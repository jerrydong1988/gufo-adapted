#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SERVE_RUNNER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SERVE_RUNNER_HPP_

// Qwen3.8-Flash-Next TextModelRunner adapter and model package registration.
//
// Implemented in serve_runner.cpp (HIP builds only). The inference backend
// installs pre-loaded models through CreateTextRunner and resolves file
// loads through the registered package.

#include <cstdint>
#include <memory>
#include <string>

namespace gufo::server {
class TextModelRunner;
}

namespace gufo::models::qwen38_flash_next {
class Model;
}

namespace gufo::models::qwen38_flash_next::serve {

[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::string artifact_fingerprint = {},
    std::string mtp_fingerprint = {});

void RegisterFlashNextPackage();

}  // namespace gufo::models::qwen38_flash_next::serve

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SERVE_RUNNER_HPP_
