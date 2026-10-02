#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::cli {

struct BenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  std::string logit_eval_path;
  std::string logit_out;
  std::string logit_schedules{"1:1,2,3,4,5,6,7,8"};
  std::string speculative_backend{""};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  sampling::SamplingConfig sampling{.seed = 0};
  bool verbose{false};
  bool generic{false};
};

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

/// Prefix tokens prefilled for generation-only generic runs (`-n` without
/// `-p`), mirroring the fixed prefix the specialized harnesses use.
inline constexpr std::size_t kGenericBenchGenOnlyPrefixTokens = 16;

/// One generic benchmark workload. `gen_len == 0` means prefill-only.
struct GenericBenchCase {
  std::size_t prompt_len{0};
  std::size_t gen_len{0};
};

/// Expands the requested prompt/generation sizes into executable workloads:
/// the cross product when both are present, prefill-only cases when only
/// `-p` is given, fixed-prefix generation-only cases when only `-n` is
/// given, and empty when no workload was requested.
[[nodiscard]] std::vector<GenericBenchCase> ExpandGenericBenchCases(
    const BenchOptions& opt);

/// Rejects options the generic path cannot honor (numerical validation,
/// context depths) before any model loads. Returns the error, if any.
[[nodiscard]] std::optional<std::string> CheckGenericBenchOptions(
    const BenchOptions& opt);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
