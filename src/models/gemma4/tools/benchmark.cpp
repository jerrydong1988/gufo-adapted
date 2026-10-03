#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/gemma4/model.hpp"
#include "src/models/gemma4/serve_runner.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "usage: gemma4_bench MODEL ASSISTANT OUTPUT_JSON PROMPT_TOKENS");
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](auto start) {
      return std::chrono::duration<double, std::milli>(Clock::now() - start)
          .count();
    };
    auto model = gufo::models::gemma4::Model::Load(argv[1], 4096);
    model->LoadAssistant(argv[2]);
    auto plain = gufo::models::gemma4::CreateTextRunner(model, 0);
    auto mtp = gufo::models::gemma4::CreateTextRunner(model, 7);
    std::string text;
    for (int i = 0; i < 300; ++i)
      text += "The event welcomes neighbors, volunteers and families. ";
    text +=
        "Produce a detailed numbered list of 200 practical tips for "
        "organizing this community event. Keep writing until item 200. "
        "Include one sentence per item.";
    const std::array messages{gufo::models::common::ChatMessage{
        gufo::models::common::ChatRole::kUser, text}};
    auto prompt = model->tokenizer().Encode(
        gufo::models::gemma4::RenderChat(messages, {}), false);
    if (prompt.size() < 2048)
      throw std::runtime_error("benchmark prompt is too short");
    // Retain BOS and the complete final instruction/assistant prefix.
    prompt.erase(prompt.begin() + 1,
                 prompt.begin() + 1 + (prompt.size() - 2048));
    std::ofstream fixture(argv[4]);
    for (const auto token : prompt)
      fixture << token << '\n';
    if (!fixture)
      throw std::runtime_error("prompt fixture write failed");
    auto state = plain->CreateState();
    for (std::size_t offset = 0; offset < 16;)
      offset += plain->Prefill(*state, std::span(prompt).first(16), offset, 16)
                    .consumed_tokens;
    for (int i = 0; i < 8; ++i)
      plain->Advance(*state, prompt[16 + i]);
    state->Invalidate();
    const auto start = Clock::now();
    for (std::size_t offset = 0; offset < prompt.size();)
      offset += plain->Prefill(*state, prompt, offset, 128).consumed_tokens;
    const auto prefill_ms = elapsed(start);
    std::cout << "prefill_ms=" << prefill_ms << std::endl;
    const auto capture_start = Clock::now();
    auto saved = plain->Snapshot(*state);
    const auto capture_ms = elapsed(capture_start);
    nlohmann::json report{{"model", argv[1]},
                          {"assistant", argv[2]},
                          {"context", 4096},
                          {"concurrency", 1},
                          {"prompt_tokens", prompt},
                          {"prefill_tokens", prompt.size()},
                          {"prefill_ms", prefill_ms},
                          {"prefill_tps", 1000 * prompt.size() / prefill_ms},
                          {"snapshot_bytes", saved->PayloadBytes()},
                          {"snapshot_ms", capture_ms},
                          {"weights_bytes", model->WeightBytes()},
                          {"kv_bytes", model->KvBytes()},
                          {"scratch_bytes", model->ScratchBytes()}};
    std::vector<std::uint32_t> baseline;
    std::vector<float> logits;
    std::uint64_t rng = 0;
    for (const bool speculative : {false, true}) {
      auto& runner = speculative ? *mtp : *plain;
      runner.RestoreOrFork(*state, *saved);
      gufo::sampling::SamplerState sampler({.seed = 42}, prompt);
      std::vector<std::uint32_t> ids;
      std::size_t drafts = 0, accepted = 0;
      const auto decode_start = Clock::now();
      while (ids.size() < 128) {
        const auto step = runner.DecodeStep(
            *state,
            std::min<std::size_t>(speculative ? 8 : 1, 128 - ids.size()),
            sampler);
        if (step.failure)
          std::rethrow_exception(step.failure);
        drafts += step.draft_tokens;
        accepted += step.draft_accepted_tokens;
        for (const auto& selection : step.selections) {
          ids.push_back(selection.token);
          sampler.Accept(selection.token);
        }
        if (step.stop)
          break;
        if (step.selections.empty())
          throw std::runtime_error("decode made no progress");
      }
      const auto ms = elapsed(decode_start);
      auto end = runner.Snapshot(*state);
      const auto row = gufo::models::gemma4::SnapshotLogits(*end);
      if (!speculative) {
        baseline = ids;
        logits.assign(row.begin(), row.end());
        rng = sampler.rng_state();
      } else if (baseline != ids || rng != sampler.rng_state() ||
                 !std::equal(logits.begin(), logits.end(), row.begin())) {
        throw std::runtime_error("AR/MTP tokens, RNG or full logits differ");
      }
      report[speculative ? "mtp" : "ar"] = {
          {"tokens", ids},
          {"completed_tokens", ids.size()},
          {"decode_ms", ms},
          {"decode_tps", 1000 * ids.size() / ms},
          {"drafts", drafts},
          {"accepted", accepted},
          {"position", runner.CheckpointPosition(*state)}};
      std::cout << (speculative ? "mtp" : "ar") << " tokens=" << ids.size()
                << " ms=" << ms << " drafts=" << drafts
                << " accepted=" << accepted << std::endl;
    }
    report["exact_ar_mtp_parity"] = true;
    std::ofstream output(argv[3]);
    output << report.dump(2) << '\n';
    if (!output)
      throw std::runtime_error("benchmark report write failed");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
