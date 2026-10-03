// Independent reference: link only the pinned llama.cpp build, never Gufo.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

struct LayerCapture {
  std::filesystem::path directory;
  int position{0};
  bool f32{false};
};

static bool Capture(ggml_tensor* tensor, bool ask, void* opaque) {
  auto& capture = *static_cast<LayerCapture*>(opaque);
  if (ask && capture.f32 && tensor->op == GGML_OP_MUL_MAT) {
    ggml_prec_set_acc(tensor, GGML_PREC_F32);
  }
  const std::string name = ggml_get_name(tensor);
  const bool wanted =
      name == "inp_scaled" || name == "result_norm" || name == "attn_norm-0" ||
      name == "Qcur_pos-0" || name == "Kcur_pos-0" || name == "Vcur_normed-0" ||
      name == "attn_out-0" || name == "l_out-0" || name == "l_out-5";
  if (ask)
    return wanted && !capture.directory.empty();
  if (wanted && !capture.directory.empty()) {
    std::vector<char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    std::ofstream output(capture.directory / (std::to_string(capture.position) +
                                              "-" + name + ".bin"),
                         std::ios::binary);
    output.write(data.data(), data.size());
  }
  return true;
}

int main(int argc, char** argv) {
  try {
    if (argc < 5)
      throw std::runtime_error(
          "usage: llama-reference MODEL TOKENS LOGITS BATCH [LAYERS] or "
          "llama-reference --tokenize MODEL TEXT TOKENS [ADD_SPECIAL]");
    const bool tokenize = std::string(argv[1]) == "--tokenize";
    llama_backend_init();
    auto parameters = llama_model_default_params();
    parameters.n_gpu_layers = tokenize ? 0 : 999;
    parameters.vocab_only = tokenize;
    auto* model =
        llama_model_load_from_file(argv[tokenize ? 2 : 1], parameters);
    if (!model)
      throw std::runtime_error("model load failed");
    const auto* vocabulary = llama_model_get_vocab(model);
    if (tokenize) {
      std::ifstream input(argv[3], std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(input)), {});
      std::vector<llama_token> ids(text.size() + 1024);
      const int size =
          llama_tokenize(vocabulary, text.data(), text.size(), ids.data(),
                         ids.size(), argc < 6 || std::stoi(argv[5]) != 0, true);
      if (size < 0)
        throw std::runtime_error("tokenization failed");
      std::ofstream output(argv[4]);
      for (int i = 0; i < size; ++i)
        output << ids[i] << '\n';
      std::cout << "tokens=" << size << '\n';
    } else {
      std::ifstream input(argv[2]);
      std::vector<llama_token> ids;
      for (llama_token id; input >> id;)
        ids.push_back(id);
      if (ids.empty())
        throw std::runtime_error("empty token fixture");
      const int width = std::stoi(argv[4]);
      if (width < 1 || width > 512)
        throw std::runtime_error("invalid batch");
      auto settings = llama_context_default_params();
      settings.n_ctx = 4096;
      settings.n_batch = settings.n_ubatch = 512;
      settings.n_threads = settings.n_threads_batch = 8;
      settings.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      settings.type_k = settings.type_v = GGML_TYPE_F32;
      settings.swa_full = true;
      LayerCapture capture;
      if (argc > 5) {
        if (std::string(argv[5]) != "-") {
          capture.directory = argv[5];
          std::filesystem::create_directories(capture.directory);
        }
        settings.cb_eval = Capture;
        settings.cb_eval_user_data = &capture;
        capture.f32 = argc > 6 && std::string(argv[6]) == "--f32";
      }
      const bool window = argc > 7 && std::string(argv[7]) == "--window";
      if (window) {
        const auto pattern = ids;
        ids.resize(2052);
        for (std::size_t i = 0; i < ids.size(); ++i)
          ids[i] = pattern[i % pattern.size()];
      }
      auto* context = llama_init_from_model(model, settings);
      if (!context)
        throw std::runtime_error("context creation failed");
      auto batch = llama_batch_init(width, 0, 1);
      std::ofstream output(argv[3], std::ios::binary);
      const int vocab = llama_vocab_n_tokens(vocabulary);
      for (std::size_t offset = 0; offset < ids.size(); offset += width) {
        capture.position = offset;
        const int real_tokens =
            std::min<std::size_t>(width, ids.size() - offset);
        // A causal suffix does not change preceding rows. Keep short final
        // batches out of quantized MMVQ when using the FP32 BLAS oracle.
        batch.n_tokens =
            capture.f32 && width > 8 ? std::max(9, real_tokens) : real_tokens;
        for (int j = 0; j < batch.n_tokens; ++j) {
          batch.token[j] = j < real_tokens ? ids[offset + j] : ids.back();
          batch.pos[j] = offset + j;
          batch.n_seq_id[j] = 1;
          batch.seq_id[j][0] = 0;
          batch.logits[j] = true;
        }
        if (llama_decode(context, batch))
          throw std::runtime_error("decode failed");
        for (int j = 0; j < real_tokens; ++j) {
          const auto position = offset + j + 1;
          if (window && position != 20 &&
              !(position >= 1023 && position <= 1028) && position < 2049)
            continue;
          output.write(
              reinterpret_cast<const char*>(llama_get_logits_ith(context, j)),
              vocab * sizeof(float));
        }
        if ((offset / width) % 64 == 0)
          std::cout << "position=" << offset << '\n';
      }
      if (!output)
        throw std::runtime_error("logit output failed");
      llama_batch_free(batch);
      llama_free(context);
      std::cout << "rows=" << ids.size() << " vocab=" << vocab << '\n';
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
