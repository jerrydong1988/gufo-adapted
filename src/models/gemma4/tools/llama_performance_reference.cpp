// Timings use the original pinned llama/ggml DLLs, with no precision overlay.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "llama.h"

static std::vector<llama_token> Read(const char* path) {
  std::ifstream input(path);
  std::vector<llama_token> ids;
  for (llama_token id; input >> id;)
    ids.push_back(id);
  if (ids.empty())
    throw std::runtime_error("empty token fixture");
  return ids;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "usage: MODEL PROMPT CONTINUATION OUTPUT_JSON");
    const auto prompt = Read(argv[2]), continuation = Read(argv[3]);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 999;
    auto* model = llama_model_load_from_file(argv[1], params);
    if (!model)
      throw std::runtime_error("reference model load failed");
    auto settings = llama_context_default_params();
    settings.n_ctx = 4096;
    settings.n_batch = settings.n_ubatch = 128;
    settings.n_threads = settings.n_threads_batch = 8;
    settings.type_k = settings.type_v = GGML_TYPE_F32;
    settings.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    settings.swa_full = true;
    auto* context = llama_init_from_model(model, settings);
    if (!context)
      throw std::runtime_error("reference context creation failed");
    auto batch = llama_batch_init(128, 0, 1);
    const auto evaluate = [&](std::span<const llama_token> tokens,
                              std::size_t position) {
      batch.n_tokens = tokens.size();
      for (std::size_t j = 0; j < tokens.size(); ++j) {
        batch.token[j] = tokens[j];
        batch.pos[j] = position + j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = j + 1 == tokens.size();
      }
      if (llama_decode(context, batch))
        throw std::runtime_error("reference decode failed");
    };
    evaluate(std::span(prompt).first(16), 0);
    for (int i = 0; i < 8; ++i)
      evaluate(std::span(prompt).subspan(16 + i, 1), 16 + i);
    llama_memory_clear(llama_get_memory(context), true);
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](auto start) {
      return std::chrono::duration<double, std::milli>(Clock::now() - start)
          .count();
    };
    const auto start = Clock::now();
    for (std::size_t offset = 0; offset < prompt.size(); offset += 128)
      evaluate(std::span(prompt).subspan(
                   offset, std::min<std::size_t>(128, prompt.size() - offset)),
               offset);
    const auto prefill = elapsed(start);
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::size_t greedy_agreement = 0;
    const auto decode_start = Clock::now();
    // Native greedy IDs fix the complete cache history for matched timings.
    for (std::size_t i = 0; i < continuation.size(); ++i) {
      const auto* logits = llama_get_logits(context);
      if (std::max_element(logits, logits + vocabulary) - logits ==
          continuation[i])
        ++greedy_agreement;
      evaluate(std::span(continuation).subspan(i, 1), prompt.size() + i);
    }
    const auto decode = elapsed(decode_start);
    std::ofstream output(argv[4]);
    output << "{\n  \"prefill_tokens\": " << prompt.size()
           << ",\n  \"prefill_ms\": " << prefill
           << ",\n  \"prefill_tps\": " << 1000 * prompt.size() / prefill
           << ",\n  \"decode_tokens\": " << continuation.size()
           << ",\n  \"decode_ms\": " << decode
           << ",\n  \"decode_tps\": " << 1000 * continuation.size() / decode
           << ",\n  \"greedy_agreement\": " << greedy_agreement
           << ",\n  \"precision\": \"standard HIP, F32 KV, flash off\"\n}\n";
    if (!output)
      throw std::runtime_error("reference report write failed");
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
