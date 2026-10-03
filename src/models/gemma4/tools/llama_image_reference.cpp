// Independent language-model image-block oracle: llama/ggml only.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

static bool Precision(ggml_tensor* tensor, bool ask, void*) {
  if (ask && tensor->op == GGML_OP_MUL_MAT)
    ggml_prec_set_acc(tensor, GGML_PREC_F32);
  return false;
}
static std::vector<llama_token> Tokens(const char* path) {
  std::ifstream input(path);
  std::vector<llama_token> result;
  for (llama_token id; input >> id;)
    result.push_back(id);
  if (result.empty())
    throw std::invalid_argument("empty token fixture");
  return result;
}
int main(int argc, char** argv) {
  try {
    if (argc != 6 && argc != 7)
      throw std::invalid_argument(
          "usage: llama-image-reference MODEL PREFIX EMBEDDINGS SUFFIX LOGITS "
          "[PREFIX_LENGTH]");
    auto prefix = Tokens(argv[2]);
    const auto suffix = Tokens(argv[4]);
    if (argc == 7) {
      const auto pattern = prefix;
      prefix.resize(std::stoul(argv[6]));
      for (std::size_t i = 0; i < prefix.size(); ++i)
        prefix[i] = pattern[i % pattern.size()];
    }
    std::ifstream image_file(argv[3], std::ios::binary | std::ios::ate);
    const auto bytes = image_file.tellg();
    if (bytes <= 0 || bytes % (5376 * sizeof(float)))
      throw std::invalid_argument("invalid embedding file");
    std::vector<float> embeddings(std::size_t(bytes) / sizeof(float));
    image_file.seekg(0);
    image_file.read(reinterpret_cast<char*>(embeddings.data()), bytes);
    const int n_image = embeddings.size() / 5376;
    if (n_image > 280 || prefix.size() + n_image + suffix.size() > 4096)
      throw std::invalid_argument("fixture exceeds bounded context");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
      throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.swa_full = true;
    cp.cb_eval = Precision;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
      throw std::runtime_error("context creation failed");
    auto text = llama_batch_init(512, 0, 1);
    std::size_t position = 0;
    const auto fill = [&](llama_batch& batch, int n) {
      batch.n_tokens = n;
      for (int i = 0; i < n; ++i) {
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
      }
    };
    for (std::size_t offset = 0; offset < prefix.size(); offset += 512) {
      const int n = std::min<std::size_t>(512, prefix.size() - offset);
      fill(text, n);
      std::copy_n(prefix.data() + offset, n, text.token);
      if (llama_decode(ctx, text))
        throw std::runtime_error("prefix decode failed");
      position += n;
    }
    auto image = llama_batch_init(n_image, 5376, 1);
    fill(image, n_image);
    std::copy(embeddings.begin(), embeddings.end(), image.embd);
    llama_set_causal_attn(ctx,
                          false);  // Gemma 31B applies this only to SWA layers.
    if (llama_decode(ctx, image))
      throw std::runtime_error("image decode failed");
    position += n_image;
    llama_set_causal_attn(ctx, true);
    std::ofstream output(argv[5], std::ios::binary);
    const auto write = [&](int row) {
      const float* logits = llama_get_logits_ith(ctx, row);
      if (!logits)
        throw std::runtime_error("missing reference logits");
      for (int i = 0; i < 262144; ++i)
        if (!std::isfinite(logits[i]))
          throw std::runtime_error("non-finite reference logits");
      output.write(reinterpret_cast<const char*>(logits),
                   262144 * sizeof(float));
    };
    write(n_image - 1);
    for (std::size_t offset = 0; offset < suffix.size(); offset += 512) {
      const int n = std::min<std::size_t>(512, suffix.size() - offset);
      fill(text, n);
      std::copy_n(suffix.data() + offset, n, text.token);
      if (llama_decode(ctx, text))
        throw std::runtime_error("suffix decode failed");
      position += n;
      for (int i = 0; i < n; ++i)
        write(i);
    }
    if (!output)
      throw std::runtime_error("logit write failed");
    llama_batch_free(image);
    llama_batch_free(text);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "image_tokens=" << n_image << " position=" << position
              << " rows=" << suffix.size() + 1 << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
