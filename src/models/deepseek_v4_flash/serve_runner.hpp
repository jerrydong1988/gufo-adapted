#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_SERVE_RUNNER_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_SERVE_RUNNER_HPP_

// DeepSeek V4 Flash TextModelRunner adapter and model package registration.
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

namespace gufo::models::deepseek_v4_flash {
class Model;
}

namespace gufo::models::deepseek_v4_flash::serve {

[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context, bool use_dspark,
    std::uint32_t max_draft_tokens, std::string artifact_fingerprint = {},
    std::string support_fingerprint = {});

void RegisterDeepSeekPackage();

}  // namespace gufo::models::deepseek_v4_flash::serve

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_SERVE_RUNNER_HPP_
