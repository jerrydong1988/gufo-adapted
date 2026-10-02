#ifndef GUFO_MODELS_COMMON_REGISTRY_HPP_
#define GUFO_MODELS_COMMON_REGISTRY_HPP_

// Process-wide registry of text model packages.
//
// Serve, prompt, and bench resolve the handling package through this registry
// instead of hard-coding architecture strings. Packages register explicitly
// via RegisterAllModelPackages() (see register_packages.hpp); there are no
// static initializers, so registration order is deterministic on every
// platform.

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/common/model_package.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::common {

class TextModelRegistry {
public:
  TextModelRegistry() = default;
  ~TextModelRegistry() = default;

  TextModelRegistry(const TextModelRegistry&) = delete;
  TextModelRegistry& operator=(const TextModelRegistry&) = delete;
  TextModelRegistry(TextModelRegistry&&) = delete;
  TextModelRegistry& operator=(TextModelRegistry&&) = delete;

  [[nodiscard]] static TextModelRegistry& Global() {
    static TextModelRegistry instance;
    return instance;
  }

  /// Registers a package. The first package claiming an architecture wins;
  /// a second fallback is rejected. Thread-safe.
  void Register(std::shared_ptr<const TextModelPackage> package) {
    if (package == nullptr)
      return;
    const std::lock_guard<std::mutex> lock(mutex_);
    if (package->IsFallback()) {
      if (fallback_ == nullptr)
        fallback_ = std::move(package);
      return;
    }
    packages_.push_back(std::move(package));
  }

  /// Finds the package handling a GGUF architecture string, or the fallback
  /// package when no package claims it (nullptr when nothing is registered).
  [[nodiscard]] const TextModelPackage* FindForArchitecture(
      std::string_view architecture) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& package : packages_) {
      for (const auto& handled : package->Architectures()) {
        if (handled == architecture)
          return package.get();
      }
    }
    return fallback_.get();
  }

  /// Reads `general.architecture` from an opened GGUF and resolves the
  /// handling package. Declared here; defined in registry.cpp so this header
  /// stays free of the GGUF reader.
  [[nodiscard]] const TextModelPackage* FindForReader(
      const core::GgufReader& reader) const;

  /// Clears all registrations. Tests only; production code registers once.
  void ResetForTesting() {
    const std::lock_guard<std::mutex> lock(mutex_);
    packages_.clear();
    fallback_.reset();
  }

  [[nodiscard]] std::vector<std::string> RegisteredNames() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& package : packages_)
      names.push_back(package->Name());
    if (fallback_ != nullptr)
      names.push_back(fallback_->Name());
    return names;
  }

private:
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<const TextModelPackage>> packages_;
  std::shared_ptr<const TextModelPackage> fallback_;
};

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_REGISTRY_HPP_
