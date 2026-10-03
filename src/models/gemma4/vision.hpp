#ifndef GUFO_MODELS_GEMMA4_VISION_HPP_
#define GUFO_MODELS_GEMMA4_VISION_HPP_
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::models::gemma4 {
inline constexpr std::size_t kMaxImageTokens = 280;
struct EncodedImage {
  core::Image pixels;
  std::vector<float> embeddings;
  [[nodiscard]] std::size_t Tokens() const { return embeddings.size() / 5376; }
};
// RGB8, 48-pixel alignment, Pillow bicubic filtering and black centered
// padding.
[[nodiscard]] core::Image PreprocessImage(const core::Image& input);
class Vision {
public:
  struct Impl;
  explicit Vision(std::unique_ptr<Impl> impl);
  ~Vision();
  [[nodiscard]] static std::unique_ptr<Vision> Load(const std::string& path);
  [[nodiscard]] EncodedImage Encode(core::Image pixels,
                                    const std::function<bool()>& cancelled = {},
                                    const std::string& directory = {});
  [[nodiscard]] std::size_t WeightBytes() const;
  [[nodiscard]] std::size_t ScratchBytes() const;
  [[nodiscard]] const std::string& Identity() const;

private:
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::gemma4
#endif
