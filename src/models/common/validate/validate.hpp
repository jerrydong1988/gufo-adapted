#ifndef GUFO_MODELS_COMMON_VALIDATE_HPP_
#define GUFO_MODELS_COMMON_VALIDATE_HPP_

// Common validation harness for text model packages, driven entirely through
// the package interface (LoadedTextModel + TextModelRunner). A new package
// gets repeatable correctness checks by loading its model and calling
// ValidateLoadedModel; failures name the first divergent check.
//
// The harness is behavioral, not numerical: it proves determinism,
// decode-path equivalence, and snapshot/restore fidelity. Numerical
// truth (logits vs an independent reference) stays model-specific.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/common/model_package.hpp"

namespace gufo::models::common {

struct ValidateOptions {
  std::vector<std::string> probe_texts{
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "What is 2+2? Reply briefly.",
  };
  /// Prompt tokens fed before decoding (probes repeat to fill it).
  std::size_t prefill_tokens{64};
  std::size_t decode_tokens{16};
  /// Greedy by default so runs must reproduce exactly.
  sampling::SamplingConfig sampling;
};

struct ValidateCheck {
  std::string name;
  bool passed{false};
  bool skipped{false};
  std::string detail;
};

struct ValidationReport {
  std::string model_id;
  std::vector<ValidateCheck> checks;
  [[nodiscard]] bool passed() const {
    for (const auto& check : checks) {
      if (!check.passed && !check.skipped) {
        return false;
      }
    }
    return true;
  }
};

/// Runs every applicable check against a loaded model. Throws only on
/// harness misuse (null runner); model failures are reported as failed
/// checks with the exception text in `detail`.
[[nodiscard]] ValidationReport ValidateLoadedModel(
    const LoadedTextModel& loaded, const ValidateOptions& options = {});

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_VALIDATE_HPP_
