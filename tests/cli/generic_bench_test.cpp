#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/common/registry.hpp"

namespace {

using namespace gufo::models::common;
using namespace gufo::server;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

class State final : public TextRunnerState {
public:
  void Invalidate() noexcept override {}
};

class PrefillRunner final : public TextModelRunner {
public:
  mutable std::size_t prefill_calls{0};
  mutable std::size_t prefilled_tokens{0};

  TextRunnerDescriptor Descriptor() const override {
    TextRunnerDescriptor descriptor;
    descriptor.model_id = "prefill-test";
    descriptor.state_abi = "prefill-test-v1";
    descriptor.max_context = 2;
    return descriptor;
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    TextRunnerResourceClaim claim;
    claim.requires_device_runtime_lock = false;
    return claim;
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {TextExecutionPlan{}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {42};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  std::string Decode(std::span<const TextRunnerToken>) const override {
    return {};
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<State>();
  }
  TextPrefillStep Prefill(TextRunnerState&,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    Expect(prompt.size() == 1 && prompt.front() == 42 && offset == 0 &&
               budget >= 1,
           "prefill receives the requested single token");
    ++prefill_calls;
    ++prefilled_tokens;
    return {1, true};
  }
  TextDecodeSelection SelectNext(TextRunnerState&,
                                 gufo::sampling::SamplerState&) const override {
    throw std::logic_error("prefill-only benchmark must not decode");
  }
  void Advance(TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("prefill-only benchmark must not advance");
  }
  std::size_t CheckpointPosition(const TextRunnerState&) const override {
    return 1;
  }
};

class PrefillPackage final : public TextModelPackage {
public:
  std::shared_ptr<PrefillRunner> runner = std::make_shared<PrefillRunner>();
  mutable std::size_t loads{0};

  std::string Name() const override { return "generic-bench-test"; }
  std::vector<std::string> Architectures() const override { return {Name()}; }
  bool ValidateTemplate(const gufo::core::GgufReader&,
                        std::string*) const override {
    return true;
  }
  std::uint32_t NativeContext(const gufo::core::GgufReader&) const override {
    return 2;
  }
  bool ValidateLoadOptions(const TextModelLoadOptions&,
                           std::string*) const override {
    return true;
  }
  std::unique_ptr<LoadedTextModel> Load(const TextModelLoadOptions& options,
                                        std::string*) const override {
    Expect(options.max_context == 2, "one-token prefill reserves two tokens");
    ++loads;
    auto loaded = std::make_unique<LoadedTextModel>();
    loaded->runner = runner;
    loaded->model_id = Name();
    loaded->max_context = 2;
    return loaded;
  }
};

template<typename T>
void Write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void WriteString(std::ostream& out, std::string_view value) {
  Write(out, static_cast<std::uint64_t>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void TestSingleTokenPrefill() {
  const auto package = std::make_shared<PrefillPackage>();
  TextModelRegistry::Global().Register(package);
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo-generic-bench-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".gguf");
  {
    // Metadata-only GGUF: dispatch is real, while the package owns the fake
    // load.
    std::ofstream out(path, std::ios::binary);
    Write(out, std::uint32_t{0x46554747});
    Write(out, std::uint32_t{3});
    Write(out, std::uint64_t{0});
    Write(out, std::uint64_t{1});
    WriteString(out, "general.architecture");
    Write(out, std::uint32_t{8});
    WriteString(out, package->Name());
    while (static_cast<std::streamoff>(out.tellp()) % 32 != 0 && out) {
      out.put('\0');
    }
    Expect(out.good(), "test GGUF was written");
  }
  const auto model_path = path.string();
  const char* args[] = {"--generic", "-m", model_path.c_str(), "-p", "1"};
  const int result = gufo::cli::RunBench(args);
  std::filesystem::remove(path);
  Expect(result == 0, "one-token prefill-only benchmark succeeds");
  Expect(package->loads == 1, "benchmark loads the registered package");
  Expect(package->runner->prefill_calls == 2 &&
             package->runner->prefilled_tokens == 2,
         "benchmark and determinism repeat each execute one prefill token");
}

}  // namespace

int main() {
  TestSingleTokenPrefill();
  std::cout << "All generic benchmark CLI tests passed.\n";
}
