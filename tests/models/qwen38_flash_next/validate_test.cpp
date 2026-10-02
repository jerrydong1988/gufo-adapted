// Package validation through the registered model package: loads weights,
// runs the common harness, and reports the first divergent check.
#include "src/models/common/validate/validate.hpp"

#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/common/register_packages.hpp"
#include "src/models/common/registry.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string draft_path;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--mtp-model" && i + 1 < argc) {
      draft_path = argv[++i];
    } else {
      std::cerr << "usage: qwen38_flash_next_validate_test --model FIRST.gguf "
                   "[--mtp-model MTP.gguf]\n";
      return 2;
    }
  }
  if (model_path.empty()) {
    return 77;
  }
  gufo::models::common::RegisterAllModelPackages();
  std::string open_error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &open_error);
  Expect(reader != nullptr, open_error);
  const gufo::models::common::TextModelPackage* package =
      gufo::models::common::TextModelRegistry::Global().FindForReader(*reader);
  Expect(package != nullptr, "no package handles this GGUF");
  std::cout << "Validating through package '" << package->Name() << "'\n";

  gufo::models::common::TextModelLoadOptions load_options;
  load_options.model_path = model_path;
  load_options.max_context = 4096;
  load_options.session_count = 1;
  if (!draft_path.empty()) {
    load_options.speculative.backend =
        gufo::models::common::SpeculativeBackend::kMtp;
    load_options.speculative.draft_model_path = draft_path;
  }
  std::string load_error;
  auto loaded = package->Load(load_options, &load_error);
  Expect(loaded != nullptr && loaded->runner != nullptr, load_error);

  const auto report = gufo::models::common::ValidateLoadedModel(*loaded);
  std::cout << "model: " << report.model_id << "\n";
  bool failed = false;
  for (const auto& check : report.checks) {
    std::cout << (check.passed    ? "PASS "
                  : check.skipped ? "SKIP "
                                  : "FAIL ")
              << check.name << ": " << check.detail << "\n";
    failed = failed || (!check.passed && !check.skipped);
  }
  if (failed || !report.passed()) {
    std::cerr << "package validation failed\n";
    return 1;
  }
  std::cout << "package validation passed\n";
  return 0;
}
