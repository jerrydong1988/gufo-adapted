#include "src/models/common/validate/validate.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"

namespace {

// Fake runner with scripted decode faults. The good mode emits a fixed
// 16-token sequence (one token per width-1 call, all at once for wide
// calls) so the harness passes; each fault mode violates one decode-step
// invariant the harness must reject.
class FakeState : public gufo::server::TextRunnerState {
public:
  void Invalidate() noexcept override {}
  std::size_t emitted{0};
  std::size_t prefilled{0};
};

enum class DecodeFault {
  kNone,
  kEmptyNoStop,
  kOverBudget,
  kEmbeddedStop,
  kEmbeddedStopWithStepStop,
};

class FakeRunner : public gufo::server::TextModelRunner {
public:
  explicit FakeRunner(DecodeFault fault) : fault_(fault) {}

  [[nodiscard]] gufo::server::TextRunnerDescriptor Descriptor() const override {
    gufo::server::TextRunnerDescriptor descriptor;
    descriptor.model_id = "fake";
    descriptor.max_context = 4096;
    return descriptor;
  }
  [[nodiscard]] gufo::server::TextRunnerResourceClaim ResourceClaim()
      const override {
    return {};
  }
  [[nodiscard]] std::vector<gufo::server::TextExecutionPlan> SupportedPlans()
      const override {
    return {gufo::server::TextExecutionPlan{}};
  }
  [[nodiscard]] std::vector<gufo::server::TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<gufo::server::TextRunnerToken> ids;
    for (const char c : text) {
      ids.push_back(static_cast<gufo::server::TextRunnerToken>(c) + 1);
    }
    return ids;
  }
  [[nodiscard]] std::optional<std::vector<gufo::server::TextRunnerToken>>
  RenderAndTokenize(const gufo::server::ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const gufo::server::TextRunnerToken>) const override {
    return "";
  }
  [[nodiscard]] std::unique_ptr<gufo::server::TextRunnerState> CreateState()
      const override {
    return std::make_unique<FakeState>();
  }
  [[nodiscard]] gufo::server::TextPrefillStep Prefill(
      gufo::server::TextRunnerState& state,
      std::span<const gufo::server::TextRunnerToken> prompt, std::size_t offset,
      std::size_t) const override {
    auto& fake = static_cast<FakeState&>(state);
    fake.prefilled = prompt.size();
    return {prompt.size() - offset, true};
  }
  [[nodiscard]] gufo::server::TextDecodeSelection SelectNext(
      gufo::server::TextRunnerState&,
      gufo::sampling::SamplerState&) const override {
    return {};
  }
  void Advance(gufo::server::TextRunnerState&,
               gufo::server::TextRunnerToken) const override {}
  [[nodiscard]] gufo::server::TextDecodeStep DecodeStep(
      gufo::server::TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState&) const override {
    auto& fake = static_cast<FakeState&>(state);
    gufo::server::TextDecodeStep step;
    switch (fault_) {
      case DecodeFault::kEmptyNoStop:
        return step;
      case DecodeFault::kOverBudget: {
        for (std::size_t i = 0; i < max_tokens + 1; ++i) {
          step.selections.push_back({false, 7, ""});
        }
        return step;
      }
      case DecodeFault::kEmbeddedStop:
      case DecodeFault::kEmbeddedStopWithStepStop:
        step.stop = fault_ == DecodeFault::kEmbeddedStopWithStepStop;
        step.selections.push_back({true, 7, ""});
        return step;
      case DecodeFault::kNone:
        break;
    }
    std::size_t n = std::min(max_tokens, kTotalTokens - fake.emitted);
    for (std::size_t i = 0; i < n; ++i) {
      step.selections.push_back(
          {false,
           static_cast<gufo::server::TextRunnerToken>(1000 + fake.emitted),
           ""});
      ++fake.emitted;
    }
    step.stop = fake.emitted >= kTotalTokens;
    return step;
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const gufo::server::TextRunnerState& state) const override {
    return static_cast<const FakeState&>(state).prefilled;
  }

private:
  static constexpr std::size_t kTotalTokens = 16;
  DecodeFault fault_;
};

const gufo::models::common::ValidateCheck* FindCheck(
    const gufo::models::common::ValidationReport& report,
    const std::string& name) {
  for (const auto& check : report.checks) {
    if (check.name == name) {
      return &check;
    }
  }
  return nullptr;
}

void TestGoodRunnerPasses() {
  gufo::models::common::LoadedTextModel loaded;
  loaded.runner = std::make_shared<FakeRunner>(DecodeFault::kNone);
  loaded.model_id = "fake";
  const auto report = gufo::models::common::ValidateLoadedModel(loaded, {});
  assert(report.passed());
  const auto* decode = FindCheck(report, "decode_single_vs_multi_token");
  assert(decode != nullptr && decode->passed);
}

void TestFault(DecodeFault fault, const std::string& fragment) {
  gufo::models::common::LoadedTextModel loaded;
  loaded.runner = std::make_shared<FakeRunner>(fault);
  loaded.model_id = "fake";
  const auto report = gufo::models::common::ValidateLoadedModel(loaded, {});
  assert(!report.passed());
  const auto* decode = FindCheck(report, "decode_single_vs_multi_token");
  assert(decode != nullptr && !decode->passed && !decode->skipped);
  assert(decode->detail.find(fragment) != std::string::npos);
}

}  // namespace

int main() {
  TestGoodRunnerPasses();
  TestFault(DecodeFault::kEmptyNoStop, "no tokens without a stop signal");
  TestFault(DecodeFault::kOverBudget, "more selections than requested");
  TestFault(DecodeFault::kEmbeddedStop, "embeds a stop selection");
  TestFault(DecodeFault::kEmbeddedStopWithStepStop, "embeds a stop selection");
  std::cout << "All validation harness tests passed.\n";
  return 0;
}
