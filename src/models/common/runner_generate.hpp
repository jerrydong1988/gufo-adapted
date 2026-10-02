#ifndef GUFO_MODELS_COMMON_RUNNER_GENERATE_HPP_
#define GUFO_MODELS_COMMON_RUNNER_GENERATE_HPP_

// Model-agnostic generation through any TextModelRunner.
//
// This is the shared engine behind generic `bench`/`prompt` support: one
// registration gives a new model working prefill/decode benchmarking and
// smoke generation without a model-specific harness. Model-specific
// harnesses remain for kernel-level benchmarks.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::server {
class TextModelRunner;
struct TextPreparedPrompt;
}  // namespace gufo::server

namespace gufo::models::common {

struct RunnerGenerateOptions {
  std::size_t max_tokens{128};
  sampling::SamplingConfig sampling;
};

struct RunnerGenerateResult {
  std::vector<std::uint32_t> tokens;
  std::size_t prefill_tokens{0};
  double prefill_ms{0.0};
  double decode_ms{0.0};
};

/// Prefills `prompt` then decodes up to `options.max_tokens` through a
/// single-state pool, stopping at the runner's stop signal. Throws on
/// model failure.
[[nodiscard]] RunnerGenerateResult GenerateWithRunner(
    std::shared_ptr<server::TextModelRunner> runner,
    std::span<const std::uint32_t> prompt,
    const RunnerGenerateOptions& options);

/// Prepared-prompt variant (chat templates, vision contexts).
[[nodiscard]] RunnerGenerateResult GenerateWithRunner(
    std::shared_ptr<server::TextModelRunner> runner,
    const server::TextPreparedPrompt& prompt,
    const RunnerGenerateOptions& options);

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_RUNNER_GENERATE_HPP_
