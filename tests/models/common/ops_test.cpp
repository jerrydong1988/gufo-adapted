// Canonical scalar reference operators: analytic correctness pins.
//
// Hand-computed and independently cross-checked values. These pin the exact
// documented formulas; model parity tests (e.g. qwen38_flash_next's
// ops_parity_test) separately prove bit-identity with model oracles.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/models/common/ops/scalar.hpp"

namespace {

using gufo::models::common::ops::L2Norm;
using gufo::models::common::ops::RmsNorm;
using gufo::models::common::ops::RmsNormUnweighted;
using gufo::models::common::ops::RopeNeox;
using gufo::models::common::ops::Sigmoid;
using gufo::models::common::ops::Silu;
using gufo::models::common::ops::Softmax;
using gufo::models::common::ops::Softplus;
using gufo::models::common::ops::SwiGLU;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                std::string_view message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << "\n";
    std::exit(1);
  }
}

void ExpectBits(float actual, float expected, std::string_view message) {
  if (std::memcmp(&actual, &expected, sizeof(float)) != 0) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << "\n";
    std::exit(1);
  }
}

void TestActivations() {
  ExpectBits(Sigmoid(0.0F), 0.5F, "Sigmoid(0) == 0.5");
  ExpectBits(Silu(0.0F), 0.0F, "Silu(0) == 0");
  ExpectBits(Silu(2.0F), 2.0F * Sigmoid(2.0F), "Silu multiply form");
  ExpectBits(Softplus(21.0F), 21.0F, "Softplus large-x identity");
  // log(2) to float precision; cross-checked against Python statistics.
  ExpectNear(Softplus(0.0F), 0.69314718F, 1e-7F, "Softplus(0) == log(2)");
  ExpectNear(Sigmoid(2.0F), 0.88079708F, 1e-7F, "Sigmoid(2)");
  ExpectNear(Silu(2.0F), 1.76159416F, 1e-6F, "Silu(2)");
}

void TestSoftmax() {
  std::vector<float> x{0.0F, 0.0F}, out(2);
  Softmax(x, out);
  ExpectBits(out[0], 0.5F, "Softmax([0,0])[0]");
  ExpectBits(out[1], 0.5F, "Softmax([0,0])[1]");
  // e^0, e^1, e^2 normalized: hand-computed in float64.
  x = {0.0F, 1.0F, 2.0F};
  out.resize(3);
  Softmax(x, out);
  ExpectNear(out[0], 0.09003057F, 1e-7F, "Softmax([0,1,2])[0]");
  ExpectNear(out[1], 0.24472847F, 1e-7F, "Softmax([0,1,2])[1]");
  ExpectNear(out[2], 0.66524096F, 1e-7F, "Softmax([0,1,2])[2]");
  // Large-logit shift invariance: max subtraction keeps this finite.
  x = {1000.0F, 1001.0F};
  out.resize(2);
  Softmax(x, out);
  ExpectNear(out[0], 0.26894142F, 1e-7F, "Softmax shifted [0]");
  ExpectNear(out[1], 0.73105858F, 1e-7F, "Softmax shifted [1]");
}

void TestSwiGLU() {
  std::vector<float> gate{0.0F, 2.0F}, up{3.0F, 4.0F}, out(2);
  SwiGLU(gate, up, out);
  ExpectBits(out[0], 0.0F, "SwiGLU zero gate");
  ExpectNear(out[1], 7.04637665F, 1e-6F, "SwiGLU(2, 4)");
}

void TestRmsNorm() {
  // Zeros stay zero: 0 * scale * gamma == 0.
  std::vector<float> x{0.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> gamma{1.0F, 2.0F, 3.0F, 4.0F};
  RmsNorm(x, gamma, 1e-6F);
  for (float v : x) {
    ExpectBits(v, 0.0F, "RmsNorm zeros");
  }
  // mean(x^2) = (1+4+9+16)/4 = 7.5; scale = 1/sqrt(7.5 + 1e-6).
  x = {1.0F, 2.0F, 3.0F, 4.0F};
  gamma = {1.0F, 1.0F, 1.0F, 1.0F};
  RmsNorm(x, gamma, 1e-6F);
  const float scale = 1.0F / std::sqrt(7.5F + 1e-6F);
  ExpectNear(x[0], 1.0F * scale, 1e-6F, "RmsNorm [1,2,3,4][0]");
  ExpectNear(x[3], 4.0F * scale, 1e-6F, "RmsNorm [1,2,3,4][3]");
  // Gamma applies per element.
  x = {1.0F, 2.0F, 3.0F, 4.0F};
  gamma = {2.0F, 1.0F, 1.0F, 0.5F};
  RmsNorm(x, gamma, 1e-6F);
  ExpectNear(x[0], 2.0F * scale, 1e-6F, "RmsNorm gamma[0]");
  ExpectNear(x[3], 2.0F * scale, 1e-6F, "RmsNorm gamma[3]");
  // Unweighted matches the all-ones weighted form bit-for-bit.
  std::vector<float> a{1.0F, -2.0F, 3.0F, -4.0F, 0.5F};
  std::vector<float> b = a;
  std::vector<float> ones(a.size(), 1.0F);
  RmsNorm(a, ones, 1e-5F);
  RmsNormUnweighted(b, 1e-5F);
  for (std::size_t i = 0; i < a.size(); ++i) {
    ExpectBits(a[i], b[i], "RmsNorm weighted ones == unweighted");
  }
}

void TestL2Norm() {
  // sum(x^2) = 9 + 16 = 25; scale = 1/sqrt(25 + eps).
  std::vector<float> x{3.0F, 4.0F};
  L2Norm(x, 1e-6F);
  const float scale = 1.0F / std::sqrt(25.0F + 1e-6F);
  ExpectNear(x[0], 3.0F * scale, 1e-6F, "L2Norm [3,4][0]");
  ExpectNear(x[1], 4.0F * scale, 1e-6F, "L2Norm [3,4][1]");
}

void TestRope() {
  // Position zero is the identity for any theta.
  std::vector<float> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const std::vector<float> original = x;
  RopeNeox(x, /*heads=*/2, /*head_dim=*/4, /*rotary_dim=*/4,
           /*pos=*/0, /*theta=*/10000.0F);
  for (std::size_t i = 0; i < x.size(); ++i) {
    ExpectBits(x[i], original[i], "RopeNeox pos 0 identity");
  }
  // Single pair rotating by exactly 1 radian: theta=1 gives freq=1 for i=0.
  x = {1.0F, 0.0F};
  RopeNeox(x, /*heads=*/1, /*head_dim=*/2, /*rotary_dim=*/2,
           /*pos=*/1, /*theta=*/1.0F);
  ExpectNear(x[0], std::cos(1.0F), 1e-6F, "RopeNeox 1 rad cos");
  ExpectNear(x[1], std::sin(1.0F), 1e-6F, "RopeNeox 1 rad sin");
  // Partial rotation leaves the tail untouched.
  x = {1.0F, 0.0F, 9.0F, 9.0F};
  RopeNeox(x, /*heads=*/1, /*head_dim=*/4, /*rotary_dim=*/2,
           /*pos=*/1, /*theta=*/1.0F);
  ExpectBits(x[2], 9.0F, "RopeNeox tail untouched [2]");
  ExpectBits(x[3], 9.0F, "RopeNeox tail untouched [3]");
}

}  // namespace

int main() {
  TestActivations();
  TestSoftmax();
  TestSwiGLU();
  TestRmsNorm();
  TestL2Norm();
  TestRope();
  std::cout << "All shared scalar op tests passed\n";
  return 0;
}
