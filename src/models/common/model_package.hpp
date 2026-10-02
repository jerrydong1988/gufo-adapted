#ifndef GUFO_MODELS_COMMON_MODEL_PACKAGE_HPP_
#define GUFO_MODELS_COMMON_MODEL_PACKAGE_HPP_

// Integration contract for a Gufo text model package.
//
// A package bundles everything serve, prompt, and bench need for one model
// family: architecture detection, chat-template validation, weight loading,
// and a TextModelRunner adapter. Registering a package (see registry.hpp)
// exposes it to every CLI without touching dispatch call sites.
//
// This header is serve-free by design: it forward-declares the runner types
// so model code never includes serving headers. Only the package
// implementation files (serve_runner.cpp) include the full runner API.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::server {
class TextModelRunner;
}

namespace gufo::models::common {

/// Speculative decoding backend requested for a load.
enum class SpeculativeBackend : std::uint8_t {
  kDisabled,
  kDFlash,
  kDSpark,
  kMtp,
};

/// Draft-length policy for adaptive speculative backends.
enum class DraftLengthPolicy : std::uint8_t {
  kFixed,
  kAdaptive,
};

/// Backend-neutral speculative request. This mirrors the serving CLI's
/// speculative options; packages validate the subset they support and reject
/// the rest with a named error.
struct SpeculativeRequest {
  SpeculativeBackend backend{SpeculativeBackend::kDisabled};
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  DraftLengthPolicy draft_policy{DraftLengthPolicy::kAdaptive};
  /// Flash-Next MTP opt-ins; all off by default. Other packages ignore them.
  bool mtp_survival{false};
  bool mtp_latin_draft_vocabulary{false};
  bool prompt_lookup{false};
};

/// Everything a package needs to load weights and build a runner.
struct TextModelLoadOptions {
  std::string model_path;
  /// Resolved serving context (non-zero; the caller resolves 0 = native via
  /// NativeContext before calling Load).
  std::uint32_t max_context{0};
  /// Requested serving concurrency. Packages that calibrate speculation or
  /// pre-allocate per-request state from this must document the coupling.
  std::size_t session_count{1};
  SpeculativeRequest speculative;
  std::string vision_model_path;
  /// Precomputed GGUF identity fingerprints (empty when disk caching is off).
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
  /// Disk-continuation-cache request. Packages validate the fingerprints
  /// against these settings and reject invalid combinations with a named
  /// error; the serving layer owns the cache itself.
  bool disk_cache_enabled{false};
  std::size_t disk_cache_capacity_bytes{0};
};

/// A loaded model ready to serve: the runner plus the serving facts the
/// HTTP layer reports.
struct LoadedTextModel {
  std::shared_ptr<server::TextModelRunner> runner;
  std::string model_id;
  bool supports_image_input{false};
  std::uint32_t max_context{0};
};

/// One model family's integration package.
class TextModelPackage {
public:
  TextModelPackage() = default;
  virtual ~TextModelPackage() = default;

  TextModelPackage(const TextModelPackage&) = delete;
  TextModelPackage& operator=(const TextModelPackage&) = delete;
  TextModelPackage(TextModelPackage&&) = delete;
  TextModelPackage& operator=(TextModelPackage&&) = delete;

  /// Stable package name (e.g. "qwen", "deepseek4", "qwen4exp").
  [[nodiscard]] virtual std::string Name() const = 0;

  /// Exact GGUF `general.architecture` values this package handles.
  [[nodiscard]] virtual std::vector<std::string> Architectures() const = 0;

  /// True for at most one package: it handles architectures no package
  /// claims explicitly. Only the fallback may return true.
  [[nodiscard]] virtual bool IsFallback() const { return false; }

  /// Validates the artifact's chat template before any weights load.
  [[nodiscard]] virtual bool ValidateTemplate(const core::GgufReader& reader,
                                              std::string* error) const = 0;

  /// Native context length from the artifact, or 0 when absent/invalid.
  [[nodiscard]] virtual std::uint32_t NativeContext(
      const core::GgufReader& reader) const = 0;

  /// Validates load options without loading weights.
  [[nodiscard]] virtual bool ValidateLoadOptions(
      const TextModelLoadOptions& options, std::string* error) const = 0;

  /// Loads weights and returns a runner ready for a TextRunnerPool.
  [[nodiscard]] virtual std::unique_ptr<LoadedTextModel> Load(
      const TextModelLoadOptions& options, std::string* error) const = 0;
};

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_MODEL_PACKAGE_HPP_
