#ifndef GUFO_MODELS_COMMON_OPS_SCALAR_HPP_
#define GUFO_MODELS_COMMON_OPS_SCALAR_HPP_

// Canonical scalar CPU reference operators shared by model packages.
//
// Every function documents its exact formula and accumulation precision.
// Sharing is only correct when the model's math matches bit-for-bit: even
// RMSNorm's epsilon placement or SiLU's divide-vs-multiply form changes
// rounding. Models adopt an operator only after the parity test below (or
// their own quality gates) prove equivalence; until then they keep their
// private oracle. See ops/README.md for the adoption progression.
//
// Adopters are recorded per operator. New models should start from these
// operators and add a genuinely new one only when their math differs.

#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::models::common::ops {

// y[i] = x[i] * (1 / sqrt(float(mean(x^2)) + eps)) * gamma[i].
//
// The mean squares accumulate in float64, narrow to float, and only then
// add epsilon (float) before the reciprocal square root. In-place.
//
// Adopters: qwen38_flash_next (bit-exact).
void RmsNorm(std::span<float> x, std::span<const float> gamma, float eps);

// Unweighted RMS normalization: y[i] = x[i] / sqrt(float(mean(x^2)) + eps).
// Bit-identical to the weighted form with an all-ones gamma on finite
// inputs. In-place.
//
// Adopters: qwen38_flash_next (bit-exact).
void RmsNormUnweighted(std::span<float> x, float eps);

// y[i] = x[i] / sqrt(float(sum(x^2)) + eps), the DeltaNet query/key
// normalization. Sum of squares accumulates in float64. In-place.
//
// Adopters: qwen38_flash_next (bit-exact).
void L2Norm(std::span<float> x, float eps);

// 1 / (1 + exp(-x)), float throughout.
//
// Adopters: qwen38_flash_next (bit-exact).
[[nodiscard]] float Sigmoid(float x) noexcept;

// x * Sigmoid(x): the multiply form. Note qwen's oracle divides instead
// (x / (1 + exp(-x))), which rounds differently; qwen has not adopted this.
//
// Adopters: qwen38_flash_next (bit-exact).
[[nodiscard]] float Silu(float x) noexcept;

// x > 20 ? x : log1p(exp(x)), float throughout.
//
// Adopters: qwen38_flash_next (bit-exact).
[[nodiscard]] float Softplus(float x) noexcept;

// out[i] = Silu(gate[i]) * up[i] with the canonical multiply-form Silu.
// qwen's SwiGLU uses its divide-form SiLU instead and has not adopted this.
//
// Adopters: (none yet; provided for new models).
void SwiGLU(std::span<const float> gate, std::span<const float> up,
            std::span<float> out);

// Numerically stable softmax: float running max, float64 exp/sum, then
// out[i] = float(exp(double(x[i] - max)) / sum). Matches qwen's
// ReferenceSoftmax bit-for-bit; qwen38_flash_next's inline MoE softmax uses
// float exp instead and keeps its own form.
//
// Adopters: (none yet; qwen-compatible).
void Softmax(std::span<const float> x, std::span<float> out);

// NEOX-style partial rotary embedding over `heads` heads of `head_dim`
// floats: within the first `rotary_dim` elements, pair (i, i + rotary_dim/2)
// rotates by pos * theta^(-2i/rotary_dim). Angles and sin/cos evaluate in
// float. Requires rotary_dim <= head_dim and even.
//
// Adopters: qwen38_flash_next (bit-exact for text IMRoPE).
void RopeNeox(std::span<float> x, std::uint32_t heads, std::uint32_t head_dim,
              std::uint32_t rotary_dim, std::uint32_t pos, float theta);

}  // namespace gufo::models::common::ops

#endif  // GUFO_MODELS_COMMON_OPS_SCALAR_HPP_
