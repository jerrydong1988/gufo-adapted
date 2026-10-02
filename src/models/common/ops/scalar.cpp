#include "src/models/common/ops/scalar.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::models::common::ops {

void RmsNorm(std::span<float> x, std::span<const float> gamma, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(ss / static_cast<double>(x.size())) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * scale * gamma[i];
  }
}

void RmsNormUnweighted(std::span<float> x, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(ss / static_cast<double>(x.size())) + eps);
  for (float& v : x) {
    v *= scale;
  }
}

void L2Norm(std::span<float> x, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(ss) + eps);
  for (float& v : x) {
    v *= scale;
  }
}

float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

float Silu(float x) noexcept {
  return x * Sigmoid(x);
}

float Softplus(float x) noexcept {
  return x > 20.0F ? x : std::log1p(std::exp(x));
}

void SwiGLU(std::span<const float> gate, std::span<const float> up,
            std::span<float> out) {
  for (std::size_t i = 0; i < gate.size(); ++i) {
    out[i] = Silu(gate[i]) * up[i];
  }
}

void Softmax(std::span<const float> x, std::span<float> out) {
  float max_val = x[0];
  for (const float v : x) {
    max_val = std::max(v, max_val);
  }
  double sum_exp = 0.0;
  for (const float val : x) {
    sum_exp += std::exp(static_cast<double>(val - max_val));
  }
  const double inv_sum = 1.0 / sum_exp;
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = static_cast<float>(std::exp(static_cast<double>(x[i] - max_val)) *
                                inv_sum);
  }
}

void RopeNeox(std::span<float> x, std::uint32_t heads, std::uint32_t head_dim,
              std::uint32_t rotary_dim, std::uint32_t pos, float theta) {
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* v = x.data() + static_cast<std::size_t>(h) * head_dim;
    for (std::uint32_t i = 0; i < half; ++i) {
      const float freq = std::pow(theta, -2.0F * static_cast<float>(i) /
                                             static_cast<float>(rotary_dim));
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float a = v[i];
      const float b = v[i + half];
      v[i] = a * c - b * s;
      v[i + half] = a * s + b * c;
    }
  }
}

}  // namespace gufo::models::common::ops
