#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "src/core/crypto/sha256.hpp"
#include "src/models/gemma4/device.hpp"
#include "src/models/gemma4/vision.hpp"

namespace gufo::models::gemma4 {
namespace {
using namespace device;
constexpr int kWidth = 1152, kFeedForward = 4304, kHeads = 16, kHeadWidth = 72;
constexpr int kPatches = kMaxImageTokens * 9;

__global__ void PackPatches(const std::uint8_t* rgb, float* patches, int width,
                            int nx, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * 768)
    return;
  const int patch = index / 768, channel = index % 768 / 256,
            pixel = index % 256;
  const int x = (patch % nx) * 16 + pixel % 16,
            y = (patch / nx) * 16 + pixel / 16;
  const float value = float(rgb[(y * width + x) * 3 + channel]) / 255.0F;
  // The pinned convolution's im2col boundary stores scaled pixels in FP16.
  patches[index] = __half2float(__float2half(2.0F * value - 1.0F));
}
__global__ void Position(float* hidden, const float* table, int nx, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count * kWidth) {
    const int patch = index / kWidth, i = index % kWidth;
    hidden[index] = (hidden[index] + table[(patch % nx) * kWidth + i]) +
                    table[(10240 + patch / nx) * kWidth + i];
  }
}
__global__ void VisionRope(float* values, int nx, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * kHeads * 36)
    return;
  const int patch = index / (kHeads * 36), head = index / 36 % kHeads,
            pair = index % 36;
  const int axis = pair / 18, i = pair % 18,
            pos = axis ? patch / nx : patch % nx;
  const float theta = pos * powf(100.0F, -2.0F * i / 36);
  const int first = patch * kWidth + head * kHeadWidth + axis * 36 + i,
            second = first + 18;
  const float a = values[first], b = values[second], c = cosf(theta),
              s = sinf(theta);
  values[first] = a * c - b * s;
  values[second] = a * s + b * c;
}
__global__ void Permute(const float* input, float* output, int count,
                        bool inverse) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * kWidth)
    return;
  const int patch = index / kWidth, head = index % kWidth / kHeadWidth,
            i = index % kHeadWidth;
  const int target = (head * count + patch) * kHeadWidth + i;
  if (inverse)
    output[index] = input[target];
  else
    output[target] = input[index];
}
__global__ void NormalizeScores(float* scores, int count) {
  __shared__ float reduction[128];
  float* row = scores + std::size_t(blockIdx.x) * count;
  const int lane = threadIdx.x;
  float maximum = -INFINITY;
  for (int i = lane; i < count; i += 128)
    maximum = fmaxf(maximum, row[i]);
  reduction[lane] = maximum;
  __syncthreads();
  for (int shift = 64; shift; shift >>= 1) {
    if (lane < shift)
      reduction[lane] = fmaxf(reduction[lane], reduction[lane + shift]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float sum = 0;
  for (int i = lane; i < count; i += 128) {
    row[i] = expf(row[i] - maximum);
    sum += row[i];
  }
  reduction[lane] = sum;
  __syncthreads();
  for (int shift = 64; shift; shift >>= 1) {
    if (lane < shift)
      reduction[lane] += reduction[lane + shift];
    __syncthreads();
  }
  const float inverse = 1.0F / reduction[0];
  for (int i = lane; i < count; i += 128)
    row[i] *= inverse;
}
__global__ void Pool(const float* input, const float* bias, const float* scale,
                     float* output, int nx, int ny) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_x = nx / 3, out_y = ny / 3;
  if (index >= out_x * out_y * kWidth)
    return;
  const int patch = index / kWidth, i = index % kWidth, x = patch % out_x * 3,
            y = patch / out_x * 3;
  float sum = 0;
  for (int dy = 0; dy < 3; ++dy)
    for (int dx = 0; dx < 3; ++dx)
      sum += input[((y + dy) * nx + x + dx) * kWidth + i];
  output[index] = (sum / 9.0F * sqrtf(float(kWidth)) - bias[i]) * scale[i];
}
}  // namespace

struct Vision::Impl {
  Stream stream;
  Blas blas{stream.value};
  std::unordered_map<std::string, Tensor> weights;
  std::string identity;
  std::size_t weight_bytes{0};
  Buffer rgb{kMaxImageTokens * 48 * 48 * 3},
      patches{kPatches * 768 * sizeof(float)},
      hidden{kPatches * kWidth * sizeof(float)},
      norm{kPatches * kWidth * sizeof(float)},
      q{kPatches * kWidth * sizeof(float)},
      k{kPatches * kWidth * sizeof(float)},
      v{kPatches * kWidth * sizeof(float)},
      pq{kPatches * kWidth * sizeof(float)},
      pk{kPatches * kWidth * sizeof(float)},
      pv{kPatches * kWidth * sizeof(float)},
      attention{kPatches * kWidth * sizeof(float)},
      update{kPatches * kWidth * sizeof(float)},
      gate{kPatches * kFeedForward * sizeof(float)},
      up{kPatches * kFeedForward * sizeof(float)},
      scores{std::size_t(kHeads) * kPatches * kPatches * sizeof(float)},
      pooled{kMaxImageTokens * kWidth * sizeof(float)},
      projected{kMaxImageTokens * 5376 * sizeof(float)};
  const Tensor& Get(const std::string& name) const { return weights.at(name); }
  void Multiply(const std::string& name, const Buffer& input, Buffer& output,
                int batch) const {
    const auto& t = Get(name);
    blas.Multiply(t.buffer.floats(), input.floats(), output.floats(), t.columns,
                  t.rows, batch);
  }
  std::size_t ScratchBytes() const {
    return rgb.bytes + patches.bytes + hidden.bytes + norm.bytes + q.bytes +
           k.bytes + v.bytes + pq.bytes + pk.bytes + pv.bytes +
           attention.bytes + update.bytes + gate.bytes + up.bytes +
           scores.bytes + pooled.bytes + projected.bytes;
  }
};

Vision::Vision(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Vision::~Vision() = default;
std::size_t Vision::WeightBytes() const {
  return impl_->weight_bytes;
}
std::size_t Vision::ScratchBytes() const {
  return impl_->ScratchBytes();
}
const std::string& Vision::Identity() const {
  return impl_->identity;
}

std::unique_ptr<Vision> Vision::Load(const std::string& path) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader)
    throw std::runtime_error(error);
  const auto require = [&](std::string_view key, std::uint32_t value) {
    if (reader->GetMetadataUint32(key) != value)
      throw std::invalid_argument("unsupported Gemma vision metadata: " +
                                  std::string(key));
  };
  if (reader->GetMetadataString("general.architecture") != "clip" ||
      reader->GetMetadataString("clip.vision.projector_type") != "gemma4v")
    throw std::invalid_argument("expected gemma4v projector");
  require("clip.vision.block_count", 27);
  require("clip.vision.embedding_length", kWidth);
  require("clip.vision.feed_forward_length", kFeedForward);
  require("clip.vision.attention.head_count", kHeads);
  require("clip.vision.patch_size", 16);
  require("clip.vision.projection_dim", 5376);
  if (reader->GetMetadataFloat32("clip.vision.attention.layer_norm_epsilon") !=
      1e-6F)
    throw std::invalid_argument("unsupported Gemma vision normalization");
  auto impl = std::make_unique<Impl>();
  std::unordered_set<std::string> expected;
  const auto load = [&](std::string name, std::vector<std::uint64_t> shape,
                        core::GgmlType type) {
    const auto* tensor = reader->FindTensor(name);
    if (!tensor || tensor->dimensions != shape || tensor->type != type)
      throw std::invalid_argument("unsupported Gemma vision tensor: " + name);
    Tensor device;
    device.columns = shape[0];
    device.rows = shape.size() > 1 ? shape[1] : 1;
    std::vector<float> converted(tensor->ElementCount());
    if (type == core::GgmlType::kF32)
      std::memcpy(converted.data(), tensor->data,
                  converted.size() * sizeof(float));
    else
      for (std::size_t i = 0; i < converted.size(); ++i) {
        std::uint16_t half;
        std::memcpy(&half, static_cast<const char*>(tensor->data) + i * 2, 2);
        const std::uint32_t bits = std::uint32_t(half) << 16;
        std::memcpy(&converted[i], &bits, 4);
      }
    device.buffer = Buffer(converted.size() * sizeof(float));
    impl->weight_bytes += device.buffer.bytes;
    Check(hipMemcpy(device.buffer.data, converted.data(), device.buffer.bytes,
                    hipMemcpyHostToDevice));
    expected.insert(name);
    impl->weights.emplace(std::move(name), std::move(device));
  };
  load("v.patch_embd.weight", {16, 16, 3, kWidth}, core::GgmlType::kF32);
  impl->weights.at("v.patch_embd.weight").columns = 768;
  impl->weights.at("v.patch_embd.weight").rows = kWidth;
  load("v.position_embd.weight", {kWidth, 10240, 2}, core::GgmlType::kF32);
  load("v.std_bias", {kWidth}, core::GgmlType::kF32);
  load("v.std_scale", {kWidth}, core::GgmlType::kF32);
  load("mm.input_projection.weight", {kWidth, 5376}, core::GgmlType::kBF16);
  for (int i = 0; i < 27; ++i) {
    const auto prefix = "v.blk." + std::to_string(i) + ".";
    for (const auto* name : {"ln1", "ln2", "attn_post_norm", "ffn_post_norm"})
      load(prefix + name + ".weight", {kWidth}, core::GgmlType::kF32);
    for (const auto* name : {"attn_q_norm", "attn_k_norm"})
      load(prefix + name + ".weight", {kHeadWidth}, core::GgmlType::kF32);
    for (const auto* name : {"attn_q", "attn_k", "attn_v", "attn_out"})
      load(prefix + name + ".weight", {kWidth, kWidth}, core::GgmlType::kBF16);
    load(prefix + "ffn_gate.weight", {kWidth, kFeedForward},
         core::GgmlType::kBF16);
    load(prefix + "ffn_up.weight", {kWidth, kFeedForward},
         core::GgmlType::kBF16);
    load(prefix + "ffn_down.weight", {kFeedForward, kWidth},
         core::GgmlType::kBF16);
  }
  if (reader->GetTensors().size() != expected.size())
    throw std::invalid_argument(
        "unexpected Gemma vision tensors or clipping scalars");
  impl->identity = crypto::Sha256Hex(
      {static_cast<const std::uint8_t*>(reader->GetData()), reader->GetSize()});
  return std::make_unique<Vision>(std::move(impl));
}

EncodedImage Vision::Encode(core::Image pixels,
                            const std::function<bool()>& cancelled,
                            const std::string& directory) {
  auto& e = *impl_;
  const auto stream = e.stream.value;
  pixels = PreprocessImage(pixels);
  const int nx = pixels.width / 16, ny = pixels.height / 16, count = nx * ny;
  const auto capture = [&](std::string name, const Buffer& data,
                           std::size_t elements) {
    if (directory.empty())
      return;
    std::vector<float> host(elements);
    Check(hipMemcpyAsync(host.data(), data.data, elements * sizeof(float),
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    std::filesystem::create_directories(directory);
    std::ofstream out(std::filesystem::path(directory) / (name + ".bin"),
                      std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()),
              host.size() * sizeof(float));
    if (!out)
      throw std::runtime_error("Gemma vision dump failed");
  };
  Check(hipMemcpyAsync(e.rgb.data, pixels.pixels.data(), pixels.pixels.size(),
                       hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(PackPatches, dim3((count * 768 + 127) / 128), dim3(128), 0,
                     stream, static_cast<const std::uint8_t*>(e.rgb.data),
                     e.patches.floats(), pixels.width, nx, count);
  e.Multiply("v.patch_embd.weight", e.patches, e.hidden, count);
  capture("inp", e.hidden, count * kWidth);
  hipLaunchKernelGGL(Position, dim3((count * kWidth + 127) / 128), dim3(128), 0,
                     stream, e.hidden.floats(),
                     e.Get("v.position_embd.weight").buffer.floats(), nx,
                     count);
  capture("pos_embd", e.hidden, count * kWidth);
  const float alpha = 1, beta = 0;
  for (int i = 0; i < 27; ++i) {
    if (cancelled && cancelled()) {
      Check(hipStreamSynchronize(stream));
      throw std::runtime_error("Gemma vision cancelled");
    }
    const auto p = "v.blk." + std::to_string(i) + ".";
    Normalize(stream, e.hidden.floats(),
              e.Get(p + "ln1.weight").buffer.floats(), e.norm.floats(), kWidth,
              count);
    if (i == 0)
      capture("layer_inp_normed-0", e.norm, count * kWidth);
    e.Multiply(p + "attn_q.weight", e.norm, e.q, count);
    e.Multiply(p + "attn_k.weight", e.norm, e.k, count);
    e.Multiply(p + "attn_v.weight", e.norm, e.v, count);
    Normalize(stream, e.q.floats(),
              e.Get(p + "attn_q_norm.weight").buffer.floats(), e.q.floats(),
              kHeadWidth, kHeads * count);
    Normalize(stream, e.k.floats(),
              e.Get(p + "attn_k_norm.weight").buffer.floats(), e.k.floats(),
              kHeadWidth, kHeads * count);
    Normalize(stream, e.v.floats(), nullptr, e.v.floats(), kHeadWidth,
              kHeads * count);
    hipLaunchKernelGGL(VisionRope, dim3((count * kHeads * 36 + 127) / 128),
                       dim3(128), 0, stream, e.q.floats(), nx, count);
    hipLaunchKernelGGL(VisionRope, dim3((count * kHeads * 36 + 127) / 128),
                       dim3(128), 0, stream, e.k.floats(), nx, count);
    if (i == 0) {
      capture("Qcur_pos-0", e.q, count * kWidth);
      capture("Kcur_pos-0", e.k, count * kWidth);
      capture("Vcur_normed-0", e.v, count * kWidth);
    }
    for (const auto pair : {std::pair{&e.q, &e.pq}, std::pair{&e.k, &e.pk},
                            std::pair{&e.v, &e.pv}})
      hipLaunchKernelGGL(Permute, dim3((count * kWidth + 127) / 128), dim3(128),
                         0, stream, pair.first->floats(), pair.second->floats(),
                         count, false);
    const auto stride = std::size_t(count) * kHeadWidth;
    CheckBlas(hipblasSgemmStridedBatched(
        e.blas.handle, HIPBLAS_OP_T, HIPBLAS_OP_N, count, count, kHeadWidth,
        &alpha, e.pk.floats(), kHeadWidth, stride, e.pq.floats(), kHeadWidth,
        stride, &beta, e.scores.floats(), count, std::size_t(count) * count,
        kHeads));
    hipLaunchKernelGGL(NormalizeScores, dim3(kHeads * count), dim3(128), 0,
                       stream, e.scores.floats(), count);
    CheckBlas(hipblasSgemmStridedBatched(
        e.blas.handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kHeadWidth, count, count,
        &alpha, e.pv.floats(), kHeadWidth, stride, e.scores.floats(), count,
        std::size_t(count) * count, &beta, e.pq.floats(), kHeadWidth, stride,
        kHeads));
    hipLaunchKernelGGL(Permute, dim3((count * kWidth + 127) / 128), dim3(128),
                       0, stream, e.pq.floats(), e.attention.floats(), count,
                       true);
    e.Multiply(p + "attn_out.weight", e.attention, e.update, count);
    if (i == 0)
      capture("attn_out-0", e.update, count * kWidth);
    Normalize(stream, e.update.floats(),
              e.Get(p + "attn_post_norm.weight").buffer.floats(),
              e.update.floats(), kWidth, count);
    Add(stream, e.hidden.floats(), e.update.floats(), count * kWidth);
    Normalize(stream, e.hidden.floats(),
              e.Get(p + "ln2.weight").buffer.floats(), e.norm.floats(), kWidth,
              count);
    if (i == 0)
      capture("ffn_inp_normed-0", e.norm, count * kWidth);
    e.Multiply(p + "ffn_gate.weight", e.norm, e.gate, count);
    e.Multiply(p + "ffn_up.weight", e.norm, e.up, count);
    if (i == 0) {
      capture("ffn_gate-0", e.gate, count * kFeedForward);
      capture("ffn_up-0", e.up, count * kFeedForward);
    }
    Gate(stream, e.gate.floats(), e.up.floats(), count * kFeedForward);
    if (i == 0)
      capture("ffn_geglu-0", e.gate, count * kFeedForward);
    e.Multiply(p + "ffn_down.weight", e.gate, e.update, count);
    if (i == 0)
      capture("ffn_out-0", e.update, count * kWidth);
    Normalize(stream, e.update.floats(),
              e.Get(p + "ffn_post_norm.weight").buffer.floats(),
              e.update.floats(), kWidth, count);
    Add(stream, e.hidden.floats(), e.update.floats(), count * kWidth);
    capture("layer_out-" + std::to_string(i), e.hidden, count * kWidth);
  }
  const int tokens = count / 9;
  hipLaunchKernelGGL(
      Pool, dim3((tokens * kWidth + 127) / 128), dim3(128), 0, stream,
      e.hidden.floats(), e.Get("v.std_bias").buffer.floats(),
      e.Get("v.std_scale").buffer.floats(), e.pooled.floats(), nx, ny);
  capture("std_scaled", e.pooled, tokens * kWidth);
  Normalize(stream, e.pooled.floats(), nullptr, e.pooled.floats(), kWidth,
            tokens);
  e.Multiply("mm.input_projection.weight", e.pooled, e.projected, tokens);
  capture("projected", e.projected, tokens * 5376);
  Check(hipGetLastError());
  EncodedImage result{std::move(pixels),
                      std::vector<float>(std::size_t(tokens) * 5376)};
  Check(hipMemcpyAsync(result.embeddings.data(), e.projected.data,
                       result.embeddings.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream));
  Check(hipStreamSynchronize(stream));
  if (!std::ranges::all_of(result.embeddings,
                           [](float x) { return std::isfinite(x); }))
    throw std::runtime_error("non-finite Gemma vision embeddings");
  return result;
}
}  // namespace gufo::models::gemma4
