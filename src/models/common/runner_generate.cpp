#include "src/models/common/runner_generate.hpp"

#include <limits>
#include <stdexcept>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::models::common {

namespace {

RunnerGenerateResult Generate(
    std::shared_ptr<server::TextModelRunner> runner,
    std::vector<std::uint32_t> prompt,
    std::shared_ptr<const server::TextPromptContext> context,
    std::size_t cache_prefix_tokens, const RunnerGenerateOptions& options) {
  using Clock = std::chrono::steady_clock;
  if (runner == nullptr) {
    throw std::invalid_argument("runner must not be null");
  }
  if (prompt.empty()) {
    throw std::invalid_argument("prompt must not be empty");
  }
  if (options.max_tokens == 0) {
    throw std::invalid_argument("max_tokens must be at least one");
  }
  RunnerGenerateResult result;
  result.prefill_tokens = prompt.size();
  server::TextRunnerPool pool(std::move(runner), /*state_count=*/1);
  auto request =
      pool.Acquire(std::move(prompt), options.sampling, {}, std::move(context),
                   /*reuse_prompt=*/true, cache_prefix_tokens);
  if (!request) {
    throw std::runtime_error("runner pool refused the request");
  }
  const auto prefill_start = Clock::now();
  while (!request.prefill_complete()) {
    request.Prefill(std::numeric_limits<std::size_t>::max());
  }
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - prefill_start)
          .count();
  const auto decode_start = Clock::now();
  while (result.tokens.size() < options.max_tokens) {
    server::TextDecodeStep step =
        request.DecodeStep(options.max_tokens - result.tokens.size());
    if (step.failure) {
      std::rethrow_exception(step.failure);
    }
    for (const auto& selection : step.selections) {
      result.tokens.push_back(selection.token);
    }
    if (step.stop) {
      break;
    }
  }
  result.decode_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
          .count();
  return result;
}

}  // namespace

RunnerGenerateResult GenerateWithRunner(
    std::shared_ptr<server::TextModelRunner> runner,
    std::span<const std::uint32_t> prompt,
    const RunnerGenerateOptions& options) {
  return Generate(std::move(runner),
                  std::vector<std::uint32_t>(prompt.begin(), prompt.end()),
                  /*context=*/{}, /*cache_prefix_tokens=*/0, options);
}

RunnerGenerateResult GenerateWithRunner(
    std::shared_ptr<server::TextModelRunner> runner,
    const server::TextPreparedPrompt& prompt,
    const RunnerGenerateOptions& options) {
  return Generate(std::move(runner), prompt.tokens, prompt.context,
                  prompt.cache_prefix_tokens, options);
}

}  // namespace gufo::models::common
