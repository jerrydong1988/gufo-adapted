// Independent Gemma assistant oracle; links only pinned llama/ggml libraries.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama-ext.h"

struct CaptureState {
  std::filesystem::path directory;
  int step{0};
};

static bool Capture(ggml_tensor* tensor, bool ask, void* opaque) {
  auto& state = *static_cast<CaptureState*>(opaque);
  if (ask && tensor->op == GGML_OP_MUL_MAT)
    ggml_prec_set_acc(tensor, GGML_PREC_F32);
  const std::string name = ggml_get_name(tensor);
  const bool wanted =
      !state.directory.empty() &&
      (name == "pre_proj" || name == "h_nextn" || name == "result_output" ||
       name.starts_with("out_scaled-"));
  if (ask)
    return wanted;
  if (wanted) {
    // Repeated query rows force the arithmetic-matched BLAS path. Each row
    // has identical inputs/position and reads the same frozen target KV.
    std::vector<float> row(tensor->ne[0]);
    ggml_backend_tensor_get(tensor, row.data(), 0, row.size() * sizeof(float));
    if (!std::all_of(row.begin(), row.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("non-finite reference assistant tensor: " +
                               name);
    std::string label =
        name.starts_with("out_scaled-") ? "l_out-" + name.substr(11) : name;
    std::ofstream output(
        state.directory / (std::to_string(state.step) + "-" + label + ".bin"),
        std::ios::binary);
    output.write(reinterpret_cast<const char*>(row.data()),
                 row.size() * sizeof(float));
  }
  return true;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "usage: llama-assistant-reference TARGET ASSISTANT TOKENS DIRECTORY");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    auto* target = llama_model_load_from_file(argv[1], mp);
    if (!target)
      throw std::runtime_error("target load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.swa_full = true;
    CaptureState capture;
    cp.cb_eval = Capture;
    cp.cb_eval_user_data = &capture;
    auto* context = llama_init_from_model(target, cp);
    if (!context)
      throw std::runtime_error("target context failed");
    llama_set_embeddings_nextn(context, true, false);
    std::ifstream input(argv[3]);
    std::vector<llama_token> tokens;
    for (llama_token t; input >> t;)
      tokens.push_back(t);
    if (tokens.size() < 9 || tokens.size() > 512)
      throw std::invalid_argument("oracle prefix must contain 9..512 tokens");
    auto* assistant = llama_model_load_from_file(argv[2], mp);
    if (!assistant)
      throw std::runtime_error("assistant load failed");
    cp.ctx_other = context;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    auto* draft = llama_init_from_model(assistant, cp);
    if (!draft)
      throw std::runtime_error("assistant context failed");
    llama_set_embeddings_nextn(draft, true, true);
    auto batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = true;
    }
    if (llama_decode(context, batch))
      throw std::runtime_error("target prefill failed");
    const float* logits = llama_get_logits_ith(context, -1);
    llama_token seed = std::max_element(logits, logits + 262144) - logits;
    const float* h = llama_get_embeddings_nextn_ith(context, tokens.size() - 1);
    std::vector<float> hidden(h, h + 5376);
    auto queries = llama_batch_init(16, 5376, 1);
    queries.token =
        static_cast<llama_token*>(std::malloc(16 * sizeof(llama_token)));
    capture.directory = argv[4];
    std::filesystem::create_directories(capture.directory);
    std::ofstream ids(capture.directory / "tokens.txt");
    ids << seed << '\n';
    for (int step = 0; step < 7; ++step) {
      capture.step = step;
      queries.n_tokens = 16;
      for (int i = 0; i < 16; ++i) {
        queries.token[i] = seed;
        queries.pos[i] = tokens.size();
        queries.n_seq_id[i] = 1;
        queries.seq_id[i][0] = 0;
        queries.logits[i] = true;
        std::copy(hidden.begin(), hidden.end(), queries.embd + i * 5376);
      }
      if (llama_decode(draft, queries))
        throw std::runtime_error("assistant decode failed");
      logits = llama_get_logits_ith(draft, 0);
      seed = std::max_element(logits, logits + 262144) - logits;
      h = llama_get_embeddings_nextn_ith(draft, 0);
      std::copy(h, h + 5376, hidden.begin());
      ids << seed << '\n';
      if (seed == 1 || seed == 50 || seed == 106)
        break;
    }
    ids.close();
    // Match upstream MTP ownership: the executable owns its extra token array,
    // whereas llama_batch_init owns all other arrays (separate Windows CRTs).
    std::free(queries.token);
    queries.token = nullptr;
    llama_batch_free(queries);
    llama_batch_free(batch);
    llama_free(draft);
    llama_free(context);
    llama_model_free(assistant);
    llama_model_free(target);
    llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
