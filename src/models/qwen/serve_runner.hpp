#ifndef GUFO_MODELS_QWEN_SERVE_RUNNER_HPP_
#define GUFO_MODELS_QWEN_SERVE_RUNNER_HPP_

// Qwen TextModelRunner adapter and model package registration.
//
// Implemented in serve_runner.cpp (HIP builds only). The inference backend
// installs pre-loaded models through CreateTextRunner/LoadDflashModel and
// resolves file loads through the registered package.

#include <cstdint>
#include <memory>
#include <string>

namespace gufo::hip {
class QwenGpuModel;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::server {
class TextModelRunner;
}

namespace gufo::speculative {
enum class DFlashDraftPolicy : std::uint32_t;
struct SpeculativeOptions;
}  // namespace gufo::speculative

namespace gufo::models::qwen::serve {

[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
    const speculative::SpeculativeOptions& speculative_options,
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {});

/// Loads a DFlash draft backend for an already-loaded target and fills the
/// serving speculative options. Shared by file loads and pre-loaded installs.
[[nodiscard]] std::shared_ptr<const hip::QwenDFlashGpuModel> LoadDflashModel(
    const std::shared_ptr<const hip::QwenGpuModel>& target,
    const std::string& draft_model_path, std::uint32_t max_draft_tokens,
    std::uint32_t min_draft_tokens, speculative::DFlashDraftPolicy policy,
    speculative::SpeculativeOptions* speculative_options, std::string* error);

void RegisterQwenPackage();

}  // namespace gufo::models::qwen::serve

#endif  // GUFO_MODELS_QWEN_SERVE_RUNNER_HPP_
