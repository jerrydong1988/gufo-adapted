#ifndef GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_
#define GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_
#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace gufo::server {
class TextModelRunner;
class TextRunnerSnapshot;
}  // namespace gufo::server
namespace gufo::models::gemma4 {
class Model;
[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::size_t draft_tokens = 7);
void RegisterGemma4Package();
[[nodiscard]] std::span<const float> SnapshotLogits(
    const server::TextRunnerSnapshot& snapshot);
[[nodiscard]] std::string SnapshotDigest(
    const server::TextRunnerSnapshot& snapshot);
}  // namespace gufo::models::gemma4
#endif  // GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_
