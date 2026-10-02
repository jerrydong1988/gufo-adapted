// Registry dispatch: packages own their architecture identity, serve and the
// CLIs resolve through the registry instead of hard-coded strings.
#include "src/models/common/registry.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

class FakePackage final : public gufo::models::common::TextModelPackage {
public:
  FakePackage(std::string name, std::vector<std::string> architectures,
              bool fallback = false)
      : name_(std::move(name)),
        architectures_(std::move(architectures)),
        fallback_(fallback) {}

  [[nodiscard]] std::string Name() const override { return name_; }
  [[nodiscard]] std::vector<std::string> Architectures() const override {
    return architectures_;
  }
  [[nodiscard]] bool IsFallback() const override { return fallback_; }
  [[nodiscard]] bool ValidateTemplate(const gufo::core::GgufReader&,
                                      std::string*) const override {
    return true;
  }
  [[nodiscard]] std::uint32_t NativeContext(
      const gufo::core::GgufReader&) const override {
    return 0;
  }
  [[nodiscard]] bool ValidateLoadOptions(
      const gufo::models::common::TextModelLoadOptions&,
      std::string*) const override {
    return true;
  }
  [[nodiscard]] std::unique_ptr<gufo::models::common::LoadedTextModel> Load(
      const gufo::models::common::TextModelLoadOptions&,
      std::string* error) const override {
    if (error != nullptr) {
      *error = "fake package cannot load";
    }
    return nullptr;
  }

private:
  std::string name_;
  std::vector<std::string> architectures_;
  bool fallback_;
};

}  // namespace

int main() {
  using gufo::models::common::TextModelRegistry;
  TextModelRegistry& registry = TextModelRegistry::Global();
  registry.ResetForTesting();

  Expect(registry.FindForArchitecture("qwen4exp") == nullptr,
         "Empty registry resolves nothing");
  Expect(registry.RegisteredNames().empty(),
         "Empty registry lists no packages");

  registry.Register(std::make_shared<FakePackage>(
      "qwen4exp", std::vector<std::string>{"qwen4exp"}));
  registry.Register(std::make_shared<FakePackage>("qwen",
                                                  std::vector<std::string>{},
                                                  /*fallback=*/true));

  const auto* flash = registry.FindForArchitecture("qwen4exp");
  Expect(flash != nullptr && flash->Name() == "qwen4exp",
         "Claimed architecture resolves to its package");
  const auto* fallback = registry.FindForArchitecture("llama");
  Expect(fallback != nullptr && fallback->Name() == "qwen",
         "Unknown architectures resolve to the fallback package");
  Expect(registry.FindForArchitecture("") == fallback,
         "Missing architecture resolves to the fallback package");

  // First registration wins; a second fallback is ignored.
  registry.Register(std::make_shared<FakePackage>(
      "qwen4exp-shadow", std::vector<std::string>{"qwen4exp"}));
  registry.Register(std::make_shared<FakePackage>("other-fallback",
                                                  std::vector<std::string>{},
                                                  /*fallback=*/true));
  Expect(registry.FindForArchitecture("qwen4exp")->Name() == "qwen4exp",
         "First package claiming an architecture wins");
  Expect(registry.FindForArchitecture("unknown")->Name() == "qwen",
         "First fallback wins");

  const auto names = registry.RegisteredNames();
  Expect(names.size() == 3, "Registry lists packages and the fallback");
  Expect(names[0] == "qwen4exp" && names[2] == "qwen",
         "Registry lists the fallback last");

  registry.ResetForTesting();
  Expect(registry.FindForArchitecture("qwen4exp") == nullptr,
         "Reset clears test registrations");

  std::cout << "All model registry tests passed\n";
  return 0;
}
