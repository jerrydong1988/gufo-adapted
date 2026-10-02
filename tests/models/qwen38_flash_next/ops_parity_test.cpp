// Flash-Next oracle parity: the model's scalar CPU operators must match the
// shared canonical operators bit-for-bit on fixed vectors. The embedded
// checksums were recorded from the model's pre-migration oracle; run with
// --print to regenerate them after an intentional oracle change.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/models/common/ops/scalar.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"

namespace {

namespace qfn_cpu = gufo::models::qwen38_flash_next::cpu;
namespace ops = gufo::models::common::ops;

std::uint64_t XorShift(std::uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<float> RandomVector(std::uint64_t& state, std::size_t n,
                                float scale = 4.0F) {
  std::vector<float> out(n);
  for (float& v : out) {
    const double unit =
        static_cast<double>(XorShift(state) >> 11) / 9007199254740992.0;
    v = static_cast<float>((unit * 2.0 - 1.0) * scale);
  }
  return out;
}

std::uint64_t Fnv1a(std::span<const float> values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (float v : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int b = 0; b < 4; ++b) {
      hash ^= (bits >> (b * 8)) & 0xFFU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

void ExpectBits(float actual, float expected, std::string_view message) {
  if (std::memcmp(&actual, &expected, sizeof(float)) != 0) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << "\n";
    std::exit(1);
  }
}

struct ParityChecksums {
  std::uint64_t rms_norm = 0;
  std::uint64_t rms_norm_unweighted = 0;
  std::uint64_t l2_norm = 0;
  std::uint64_t activations = 0;
  std::uint64_t rope = 0;
};

// Recorded from the model's pre-migration oracle (fixed seed 0x12345).
// Regenerate with --print after an intentional oracle change.
constexpr ParityChecksums kExpected{
    .rms_norm = 7355469231323329239ULL,
    .rms_norm_unweighted = 13931217858173322839ULL,
    .l2_norm = 8480847733680914700ULL,
    .activations = 11538760108915581556ULL,
    .rope = 7047020137096764862ULL,
};

ParityChecksums RunParity(bool print) {
  ParityChecksums sums;
  std::uint64_t state = 0x12345;
  {
    auto x = RandomVector(state, 257);
    auto y = x;
    const auto gamma = RandomVector(state, 257, 1.0F);
    qfn_cpu::RmsNorm(x, gamma.data(), 1e-6F);
    ops::RmsNorm(y, gamma, 1e-6F);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ExpectBits(y[i], x[i], "RmsNorm parity");
    }
    sums.rms_norm = Fnv1a(x);
  }
  {
    auto x = RandomVector(state, 129);
    auto y = x;
    qfn_cpu::RmsNorm(x, nullptr, 1e-5F);
    ops::RmsNormUnweighted(y, 1e-5F);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ExpectBits(y[i], x[i], "RmsNorm unweighted parity");
    }
    sums.rms_norm_unweighted = Fnv1a(x);
  }
  {
    auto x = RandomVector(state, 65);
    auto y = x;
    qfn_cpu::L2Norm(x, 1e-6F);
    ops::L2Norm(y, 1e-6F);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ExpectBits(y[i], x[i], "L2Norm parity");
    }
    sums.l2_norm = Fnv1a(x);
  }
  {
    const auto inputs = RandomVector(state, 513, 8.0F);
    std::vector<float> a;
    a.reserve(inputs.size() * 3);
    for (float v : inputs) {
      ExpectBits(ops::Sigmoid(v), qfn_cpu::Sigmoid(v), "Sigmoid parity");
      ExpectBits(ops::Silu(v), qfn_cpu::Silu(v), "Silu parity");
      ExpectBits(ops::Softplus(v), qfn_cpu::Softplus(v), "Softplus parity");
      a.push_back(qfn_cpu::Sigmoid(v));
      a.push_back(qfn_cpu::Silu(v));
      a.push_back(qfn_cpu::Softplus(v));
    }
    sums.activations = Fnv1a(a);
  }
  {
    auto x = RandomVector(state, 8 * 128);
    auto y = x;
    qfn_cpu::Rope(x.data(), /*heads=*/8, /*head_dim=*/128,
                  /*rotary_dim=*/64, /*pos=*/137, /*theta=*/100000.0F);
    ops::RopeNeox(y, /*heads=*/8, /*head_dim=*/128, /*rotary_dim=*/64,
                  /*pos=*/137, /*theta=*/100000.0F);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ExpectBits(y[i], x[i], "Rope parity");
    }
    sums.rope = Fnv1a(x);
  }
  if (print) {
    std::printf(
        "rms_norm=%llu\nrms_norm_unweighted=%llu\nl2_norm=%llu\n"
        "activations=%llu\nrope=%llu\n",
        static_cast<unsigned long long>(sums.rms_norm),
        static_cast<unsigned long long>(sums.rms_norm_unweighted),
        static_cast<unsigned long long>(sums.l2_norm),
        static_cast<unsigned long long>(sums.activations),
        static_cast<unsigned long long>(sums.rope));
  }
  return sums;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--print") {
    RunParity(/*print=*/true);
    return 0;
  }
  const ParityChecksums sums = RunParity(/*print=*/false);
  if (sums.rms_norm != kExpected.rms_norm ||
      sums.rms_norm_unweighted != kExpected.rms_norm_unweighted ||
      sums.l2_norm != kExpected.l2_norm ||
      sums.activations != kExpected.activations ||
      sums.rope != kExpected.rope) {
    std::cerr << "Oracle checksums changed; rerun with --print to inspect\n";
    return 1;
  }
  std::cout << "All Flash-Next oracle parity checks passed\n";
  return 0;
}
