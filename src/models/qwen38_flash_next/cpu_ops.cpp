#include "src/models/qwen38_flash_next/cpu_ops.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/common/ops/scalar.hpp"

namespace gufo::models::qwen38_flash_next::cpu {
namespace {

using core::GgmlType;

struct BlockQ5_1 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ5_1) == 24);

void DequantizeQ5_1(const std::uint8_t* src, float* dst, std::size_t k) {
  for (std::size_t b = 0; b < k / 32; ++b) {
    BlockQ5_1 blk;
    std::memcpy(&blk, src + b * sizeof(BlockQ5_1), sizeof(BlockQ5_1));
    const float d = gufo::quant::Fp16ToFloat(blk.d);
    const float m = gufo::quant::Fp16ToFloat(blk.m);
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      const int xh0 = ((blk.qh >> j) & 1U) << 4;
      const int xh1 = ((blk.qh >> (j + 16)) & 1U) << 4;
      y[j] = static_cast<float>((blk.qs[j] & 0x0F) | xh0) * d + m;
      y[j + 16] = static_cast<float>((blk.qs[j] >> 4) | xh1) * d + m;
    }
  }
}

}  // namespace

void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  const std::size_t k = t.cols;
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = gufo::quant::Fp16ToFloat(h);
      }
      break;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kQ5_1:
      DequantizeQ5_1(src, out, k);
      break;
    case GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(src, out, k);
      break;
    case GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, out, k);
      break;
    case GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(src, out, k);
      break;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, k);
      break;
    case GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, out, k);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, k);
      break;
    default:
      std::memset(out, 0, k * sizeof(float));
      break;
  }
}

float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x) {
  std::vector<float> w(t.cols);
  DequantizeRow(t, e, row, w.data());
  double acc = 0.0;
  for (std::size_t i = 0; i < t.cols; ++i) {
    acc += static_cast<double>(w[i]) * x[i];
  }
  return static_cast<float>(acc);
}

void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y) {
  const auto rows = static_cast<std::int64_t>(t.rows);
#pragma omp parallel
  {
    std::vector<float> w(t.cols);
#pragma omp for schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      DequantizeRow(t, e, static_cast<std::uint64_t>(r), w.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < t.cols; ++i) {
        acc += static_cast<double>(w[i]) * x[i];
      }
      y[static_cast<std::size_t>(r)] = static_cast<float>(acc);
    }
  }
}

void RmsNorm(std::span<float> x, const float* w, float eps) {
  if (w == nullptr) {
    common::ops::RmsNormUnweighted(x, eps);
  } else {
    common::ops::RmsNorm(x, {w, x.size()}, eps);
  }
}

void L2Norm(std::span<float> x, float eps) {
  common::ops::L2Norm(x, eps);
}

float Sigmoid(float x) noexcept {
  return common::ops::Sigmoid(x);
}
float Silu(float x) noexcept {
  return common::ops::Silu(x);
}
float Softplus(float x) noexcept {
  return common::ops::Softplus(x);
}

void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta) {
  common::ops::RopeNeox({x, static_cast<std::size_t>(heads) * head_dim}, heads,
                        head_dim, rotary_dim, pos, theta);
}

}  // namespace gufo::models::qwen38_flash_next::cpu
