#include "src/models/common/validate/validate.hpp"

#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::models::common {

namespace {

using server::TextDecodeStep;
using server::TextModelRunner;
using server::TextPreparedPrompt;
using server::TextRunnerState;
using server::TextRunnerToken;

std::vector<TextRunnerToken> BuildPrompt(TextModelRunner& runner,
                                         const ValidateOptions& options) {
  std::vector<TextRunnerToken> prompt;
  for (const auto& text : options.probe_texts) {
    const auto ids = runner.Tokenize(text);
    prompt.insert(prompt.end(), ids.begin(), ids.end());
    if (prompt.size() >= options.prefill_tokens) {
      break;
    }
  }
  if (prompt.empty()) {
    throw std::runtime_error("validation probes tokenized to nothing");
  }
  while (prompt.size() < options.prefill_tokens) {
    const std::size_t head = prompt.size();
    for (std::size_t i = 0; i < head && prompt.size() < options.prefill_tokens;
         ++i) {
      prompt.push_back(prompt[i]);
    }
  }
  prompt.resize(options.prefill_tokens);
  return prompt;
}

void PrefillAll(TextModelRunner& runner, TextRunnerState& state,
                std::span<const TextRunnerToken> prompt) {
  std::size_t offset = 0;
  while (offset < prompt.size()) {
    const auto step = runner.Prefill(state, prompt, offset,
                                     std::numeric_limits<std::size_t>::max());
    if (step.consumed_tokens == 0) {
      throw std::runtime_error("validation prefill made no progress");
    }
    offset += step.consumed_tokens;
    if (offset < prompt.size() && step.decode_ready) {
      break;
    }
  }
  if (offset < prompt.size()) {
    throw std::runtime_error("validation prefill stopped early");
  }
}

// Enforces the decode-step contract the scheduler relies on: a step must
// either commit tokens (within the requested budget), fail, or stop. An
// empty selection list without a stop signal means the decoder made no
// progress, and an embedded per-selection stop without a step stop would
// let callers decode past the stop. Violations throw so the harness
// reports the check as failed instead of passing on empty sequences.
void AppendDecodeSelections(const TextDecodeStep& step, std::size_t requested,
                            std::vector<TextRunnerToken>& tokens) {
  if (step.selections.size() > requested) {
    throw std::runtime_error(
        "decode step returned more selections than "
        "requested");
  }
  if (step.selections.empty() && !step.stop) {
    throw std::runtime_error(
        "decode step returned no tokens without a stop signal");
  }
  for (const auto& selection : step.selections) {
    if (selection.stop && !step.stop) {
      throw std::runtime_error(
          "decode step embeds a stop selection without stopping");
    }
    tokens.push_back(selection.token);
  }
}

std::vector<TextRunnerToken> DecodeLoop(TextModelRunner& runner,
                                        TextRunnerState& state,
                                        const sampling::SamplingConfig& config,
                                        std::size_t max_tokens) {
  std::vector<TextRunnerToken> tokens;
  sampling::SamplerState sampler(config);
  while (tokens.size() < max_tokens) {
    TextDecodeStep step =
        runner.DecodeStep(state, max_tokens - tokens.size(), sampler);
    if (step.failure) {
      std::rethrow_exception(step.failure);
    }
    // Both callers use greedy sampling; the sequences must match exactly.
    AppendDecodeSelections(step, max_tokens - tokens.size(), tokens);
    if (step.stop) {
      break;
    }
  }
  return tokens;
}

std::string JoinIds(const std::vector<TextRunnerToken>& tokens) {
  std::string out;
  for (std::size_t i = 0; i < tokens.size() && i < 8; ++i) {
    if (i > 0) {
      out += ' ';
    }
    out += std::to_string(tokens[i]);
  }
  if (tokens.size() > 8) {
    out += " ...";
  }
  return out;
}

ValidateCheck CheckTokenize(TextModelRunner& runner,
                            const ValidateOptions& options) {
  ValidateCheck check{.name = "tokenize_deterministic"};
  try {
    for (const auto& text : options.probe_texts) {
      const auto first = runner.Tokenize(text);
      const auto second = runner.Tokenize(text);
      if (first != second) {
        check.detail = "tokenization is not deterministic";
        return check;
      }
      if (!text.empty() && first.empty()) {
        check.detail = "non-empty probe produced no tokens";
        return check;
      }
    }
    check.passed = true;
    check.detail = "probes tokenize deterministically";
  } catch (const std::exception& exception) {
    check.detail = exception.what();
  }
  return check;
}

ValidateCheck CheckRender(TextModelRunner& runner,
                          const ValidateOptions& options) {
  ValidateCheck check{.name = "render_and_tokenize"};
  try {
    server::ChatRequest request;
    request.messages.emplace_back(models::common::ChatRole::kUser,
                                  options.probe_texts.front());
    const auto tokens = runner.RenderAndTokenize(request);
    if (!tokens) {
      check.skipped = true;
      check.detail = "runner does not render chat requests";
      return check;
    }
    if (tokens->empty()) {
      check.detail = "rendered prompt is empty";
      return check;
    }
    check.passed = true;
    check.detail = "rendered " + std::to_string(tokens->size()) + " tokens";
  } catch (const std::exception& exception) {
    check.detail = exception.what();
  }
  return check;
}

ValidateCheck CheckDecodeEquivalence(TextModelRunner& runner,
                                     const ValidateOptions& options) {
  ValidateCheck check{.name = "decode_single_vs_multi_token"};
  try {
    const auto prompt = BuildPrompt(runner, options);
    auto single = runner.CreateState();
    PrefillAll(runner, *single, prompt);
    // Force the single-step path one token at a time by capping the pool
    // width: DecodeStep(1) must equal one DecodeStep(N) for greedy output.
    std::vector<TextRunnerToken> single_tokens;
    {
      sampling::SamplerState sampler(options.sampling);
      while (single_tokens.size() < options.decode_tokens) {
        TextDecodeStep step = runner.DecodeStep(*single, 1, sampler);
        if (step.failure) {
          std::rethrow_exception(step.failure);
        }
        AppendDecodeSelections(step, 1, single_tokens);
        if (step.stop) {
          break;
        }
      }
    }
    auto multi = runner.CreateState();
    PrefillAll(runner, *multi, prompt);
    // Wide calls may still return one speculative cycle per call; loop like
    // the scheduler does until the budget fills or the model stops.
    std::vector<TextRunnerToken> multi_tokens;
    {
      sampling::SamplerState sampler(options.sampling);
      while (multi_tokens.size() < options.decode_tokens) {
        TextDecodeStep wide = runner.DecodeStep(
            *multi, options.decode_tokens - multi_tokens.size(), sampler);
        if (wide.failure) {
          std::rethrow_exception(wide.failure);
        }
        AppendDecodeSelections(
            wide, options.decode_tokens - multi_tokens.size(), multi_tokens);
        if (wide.stop) {
          break;
        }
      }
    }
    if (single_tokens != multi_tokens) {
      check.detail = "single-step [" + JoinIds(single_tokens) +
                     "] != multi-token [" + JoinIds(multi_tokens) + "]";
      return check;
    }
    check.passed = true;
    check.detail = std::to_string(single_tokens.size()) +
                   " tokens identical on both decode paths";
  } catch (const std::exception& exception) {
    check.detail = exception.what();
  }
  return check;
}

ValidateCheck CheckSnapshotRestore(TextModelRunner& runner,
                                   const ValidateOptions& options) {
  ValidateCheck check{.name = "snapshot_restore_fidelity"};
  if (!runner.Descriptor().capabilities.snapshot) {
    check.skipped = true;
    check.detail = "runner does not advertise snapshots";
    return check;
  }
  try {
    const auto prompt = BuildPrompt(runner, options);
    auto live = runner.CreateState();
    PrefillAll(runner, *live, prompt);
    const std::size_t position = runner.CheckpointPosition(*live);
    auto snapshot = runner.Snapshot(*live);
    if (snapshot == nullptr) {
      check.detail = "snapshot returned null";
      return check;
    }
    const auto first =
        DecodeLoop(runner, *live, options.sampling, options.decode_tokens);
    auto restored = runner.CreateState();
    runner.RestoreOrFork(*restored, *snapshot);
    runner.PreparePrefixReuse(*restored,
                              std::span<const TextRunnerToken>(prompt).first(
                                  std::min(position, prompt.size())));
    const auto second =
        DecodeLoop(runner, *restored, options.sampling, options.decode_tokens);
    if (first != second) {
      check.detail = "post-restore [" + JoinIds(second) +
                     "] != pre-snapshot [" + JoinIds(first) + "]";
      return check;
    }
    check.passed = true;
    check.detail = std::to_string(first.size()) +
                   " tokens identical across snapshot restore at position " +
                   std::to_string(position);
  } catch (const std::exception& exception) {
    check.detail = exception.what();
  }
  return check;
}

}  // namespace

ValidationReport ValidateLoadedModel(const LoadedTextModel& loaded,
                                     const ValidateOptions& options) {
  if (loaded.runner == nullptr) {
    throw std::invalid_argument("loaded model has no runner");
  }
  ValidationReport report;
  report.model_id = loaded.model_id;
  report.checks.push_back(CheckTokenize(*loaded.runner, options));
  report.checks.push_back(CheckRender(*loaded.runner, options));
  report.checks.push_back(CheckDecodeEquivalence(*loaded.runner, options));
  report.checks.push_back(CheckSnapshotRestore(*loaded.runner, options));
  return report;
}

}  // namespace gufo::models::common
