#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/gemma4/vision.hpp"

namespace gufo::models::gemma4 {
namespace {
struct Filter {
  int begin;
  std::vector<std::int32_t> weights;
};
std::vector<Filter> Filters(int input, int output) {
  std::vector<Filter> result(output);
  const double scale = double(input) / output, stretch = std::max(1.0, scale);
  const double support = 2 * stretch;
  for (int x = 0; x < output; ++x) {
    const double center = (x + 0.5) * scale;
    auto& f = result[x];
    f.begin = std::max(0, int(center - support + 0.5));
    const int end = std::min(input, int(center + support + 0.5));
    std::vector<double> values;
    double total = 0;
    for (int i = f.begin; i < end; ++i) {
      const double distance = std::abs((i - center + 0.5) / stretch);
      const double weight =
          distance < 1   ? (1.5 * distance - 2.5) * distance * distance + 1
          : distance < 2 ? ((distance - 5) * distance + 8) * distance * -0.5 + 2
                         : 0;
      values.push_back(weight);
      total += weight;
    }
    for (const double value : values) {
      const double normalized = value / total;
      f.weights.push_back(static_cast<std::int32_t>(
          normalized * (1 << 22) + (normalized < 0 ? -0.5 : 0.5)));
    }
  }
  return result;
}
core::Image Resize(const core::Image& input, int width, int height) {
  if (width == int(input.width) && height == int(input.height))
    return input;
  const auto horizontal = Filters(input.width, width),
             vertical = Filters(input.height, height);
  std::vector<std::uint8_t> temp(std::size_t(width) * input.height * 3);
  core::Image result{
      std::uint32_t(width), std::uint32_t(height),
      std::vector<std::uint8_t>(std::size_t(width) * height * 3)};
  for (std::size_t y = 0; y < input.height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        const auto& f = horizontal[x];
        std::int64_t sum = 1 << 21;
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += input.pixels[(y * input.width + f.begin + k) * 3 + c] *
                 std::int64_t(f.weights[k]);
        temp[(y * width + x) * 3 + c] =
            std::clamp<std::int64_t>(sum >> 22, 0, 255);
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        const auto& f = vertical[y];
        std::int64_t sum = 1 << 21;
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += temp[((f.begin + k) * width + x) * 3 + c] *
                 std::int64_t(f.weights[k]);
        result.pixels[(std::size_t(y) * width + x) * 3 + c] =
            std::clamp<std::int64_t>(sum >> 22, 0, 255);
      }
  return result;
}
}  // namespace

core::Image PreprocessImage(const core::Image& input) {
  if (!input.width || !input.height ||
      std::uint64_t(input.width) * input.height > core::kMaxImagePixels ||
      input.pixels.size() != std::size_t(input.width) * input.height * 3)
    throw std::invalid_argument("invalid Gemma RGB image");
  constexpr int factor = 48, minimum = 70 * factor * factor,
                maximum = kMaxImageTokens * factor * factor;
  const float width = input.width, height = input.height;
  int w = std::max(factor, int(std::round(width / factor)) * factor);
  int h = std::max(factor, int(std::round(height / factor)) * factor);
  if (std::int64_t(w) * h > maximum) {
    const float beta = std::sqrt(height * width / maximum);
    w = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    h = std::max(factor, int(std::floor(height / beta / factor)) * factor);
  } else if (std::int64_t(w) * h < minimum) {
    const float beta = std::sqrt(minimum / (height * width));
    w = int(std::ceil(width * beta / factor)) * factor;
    h = int(std::ceil(height * beta / factor)) * factor;
  }
  if (std::size_t(w / factor) * (h / factor) > kMaxImageTokens ||
      w / 16 >= 10240 || h / 16 >= 10240)
    throw std::invalid_argument(
        "Gemma image aspect ratio exceeds the bounded encoder");
  // Pinned mtmd uses PAD_CEIL, centered black padding, and RGB8 filtering.
  const float scale = std::min(w / width, h / height);
  const int content_w = std::min(w, int(std::ceil(width * scale))),
            content_h = std::min(h, int(std::ceil(height * scale)));
  const auto content = Resize(input, content_w, content_h);
  core::Image canvas{std::uint32_t(w), std::uint32_t(h),
                     std::vector<std::uint8_t>(std::size_t(w) * h * 3)};
  const int dx = (w - content_w) / 2, dy = (h - content_h) / 2;
  for (int y = 0; y < content_h; ++y)
    std::copy_n(content.pixels.data() + std::size_t(y) * content_w * 3,
                content_w * 3,
                canvas.pixels.data() + (std::size_t(y + dy) * w + dx) * 3);
  return canvas;
}
}  // namespace gufo::models::gemma4
