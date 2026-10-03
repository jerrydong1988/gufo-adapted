#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/gemma4/device.hpp"
#include "src/models/gemma4/model.hpp"
#include "src/models/gemma4/vision.hpp"

namespace gufo::models::gemma4 {
namespace {

using namespace device;

__device__ float Half(std::uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

template<int Type>
__device__ float Weight(const void* pointer, std::size_t index) {
  if constexpr (Type == 0)
    return static_cast<const float*>(pointer)[index];
  if constexpr (Type == 2) {
    const auto* bytes =
        static_cast<const unsigned char*>(pointer) + (index / 32) * 18;
    const int slot = index % 32;
    const int quantized = (bytes[2 + slot % 16] >> (slot / 16 * 4)) & 15;
    return Half(*reinterpret_cast<const std::uint16_t*>(bytes)) *
           (quantized - 8);
  }
  if constexpr (Type == 8) {
    const auto* bytes =
        static_cast<const unsigned char*>(pointer) + (index / 32) * 34;
    return Half(*reinterpret_cast<const std::uint16_t*>(bytes)) *
           reinterpret_cast<const std::int8_t*>(bytes + 2)[index % 32];
  }
  if constexpr (Type == 12 || Type == 13) {
    using Block =
        std::conditional_t<Type == 12, quant::block_q4_K, quant::block_q5_K>;
    const auto& block = static_cast<const Block*>(pointer)[index / 256];
    const int slot = index % 256, group = slot / 32;
    const int scale = group < 4 ? block.scales[group] & 63
                                : (block.scales[group + 4] & 15) |
                                      ((block.scales[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? block.scales[group + 4] & 63
                                  : (block.scales[group + 4] >> 4) |
                                        ((block.scales[group] >> 6) << 4);
    const int byte = slot / 64 * 32 + slot % 32;
    int q = (block.qs[byte] >> ((group & 1) * 4)) & 15;
    if constexpr (Type == 13)
      q |= ((block.qh[slot % 32] >> group) & 1) << 4;
    return Half(block.d) * scale * q - Half(block.dmin) * minimum;
  }
  if constexpr (Type == 14) {
    const auto& block =
        static_cast<const quant::block_q6_K*>(pointer)[index / 256];
    const int slot = index % 256, half = slot / 128, group = slot % 128 / 32;
    const int ql = (block.ql[half * 64 + (group & 1) * 32 + slot % 32] >>
                    ((group / 2) * 4)) &
                   15;
    const int qh = (block.qh[half * 32 + slot % 32] >> (group * 2)) & 3;
    return Half(block.d) * block.scales[slot / 16] * ((ql | (qh << 4)) - 32);
  }
}

template<int Type>
__global__ void MatrixVector(const void* weights, const float* input,
                             float* output, int columns, int rows) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  float sum = 0;
  for (int i = lane; i < columns; i += 32)
    sum = fmaf(Weight<Type>(weights, std::size_t(row) * columns + i), input[i],
               sum);
  for (int shift = 16; shift; shift >>= 1)
    sum += __shfl_down(sum, shift, 32);
  if (lane == 0)
    output[row] = sum;
}

template<int Type>
__global__ void DequantizeMatrix(const void* input, float* output,
                                 std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Weight<Type>(input, index);
}

void MatGemm(const Tensor& tensor, const float* input, float* output, int batch,
             const Blas& blas, Buffer& scratch, hipStream_t stream) {
  const auto count = std::size_t(tensor.columns) * tensor.rows;
  const dim3 grid((count + 127) / 128), block(128);
  if (count * sizeof(float) > scratch.bytes)
    throw std::logic_error("Gemma GEMM weight scratch exceeded");
  const float* weights = tensor.buffer.floats();
  switch (tensor.type) {
    case core::GgmlType::kF32:
      break;
    case core::GgmlType::kQ4_0:
      hipLaunchKernelGGL(DequantizeMatrix<2>, grid, block, 0, stream,
                         tensor.buffer.data, scratch.floats(), count);
      weights = scratch.floats();
      break;
    case core::GgmlType::kQ4_K:
      hipLaunchKernelGGL(DequantizeMatrix<12>, grid, block, 0, stream,
                         tensor.buffer.data, scratch.floats(), count);
      weights = scratch.floats();
      break;
    case core::GgmlType::kQ5_K:
      hipLaunchKernelGGL(DequantizeMatrix<13>, grid, block, 0, stream,
                         tensor.buffer.data, scratch.floats(), count);
      weights = scratch.floats();
      break;
    case core::GgmlType::kQ6_K:
      hipLaunchKernelGGL(DequantizeMatrix<14>, grid, block, 0, stream,
                         tensor.buffer.data, scratch.floats(), count);
      weights = scratch.floats();
      break;
    default:
      throw std::invalid_argument("unsupported Gemma GEMM storage");
  }
  blas.Multiply(weights, input, output, tensor.columns, tensor.rows, batch);
}

template<int Type>
__global__ void Embedding(const void* weights, float* output, int columns,
                          int token, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < columns)
    output[i] = Weight<Type>(weights, std::size_t(token) * columns + i) * scale;
}

void MatVec(const Tensor& tensor, const float* input, float* output,
            hipStream_t stream) {
  const dim3 grid((tensor.rows + 3) / 4), block(128);
  switch (tensor.type) {
    case core::GgmlType::kF32:
      hipLaunchKernelGGL(MatrixVector<0>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    case core::GgmlType::kQ8_0:
      hipLaunchKernelGGL(MatrixVector<8>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    case core::GgmlType::kQ4_0:
      hipLaunchKernelGGL(MatrixVector<2>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    case core::GgmlType::kQ4_K:
      hipLaunchKernelGGL(MatrixVector<12>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    case core::GgmlType::kQ5_K:
      hipLaunchKernelGGL(MatrixVector<13>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    case core::GgmlType::kQ6_K:
      hipLaunchKernelGGL(MatrixVector<14>, grid, block, 0, stream,
                         tensor.buffer.data, input, output, tensor.columns,
                         tensor.rows);
      break;
    default:
      throw std::invalid_argument("unsupported Gemma matrix storage");
  }
}

void Embed(const Tensor& tensor, std::uint32_t token, float* output,
           float scale, hipStream_t stream) {
  const dim3 grid((tensor.columns + 127) / 128), block(128);
  switch (tensor.type) {
    case core::GgmlType::kF32:
      hipLaunchKernelGGL(Embedding<0>, grid, block, 0, stream,
                         tensor.buffer.data, output, tensor.columns, token,
                         scale);
      break;
    case core::GgmlType::kQ4_0:
      hipLaunchKernelGGL(Embedding<2>, grid, block, 0, stream,
                         tensor.buffer.data, output, tensor.columns, token,
                         scale);
      break;
    case core::GgmlType::kQ4_K:
      hipLaunchKernelGGL(Embedding<12>, grid, block, 0, stream,
                         tensor.buffer.data, output, tensor.columns, token,
                         scale);
      break;
    case core::GgmlType::kQ5_K:
      hipLaunchKernelGGL(Embedding<13>, grid, block, 0, stream,
                         tensor.buffer.data, output, tensor.columns, token,
                         scale);
      break;
    case core::GgmlType::kQ6_K:
      hipLaunchKernelGGL(Embedding<14>, grid, block, 0, stream,
                         tensor.buffer.data, output, tensor.columns, token,
                         scale);
      break;
    default:
      throw std::invalid_argument("unsupported Gemma embedding storage");
  }
}

__device__ float ReduceSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  return shared[0];
}

__global__ void RmsNorm(const float* input, const float* weight, float* output,
                        int width, float epsilon) {
  __shared__ float shared[128];
  const int row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += 128) {
    const float x = input[row * width + i];
    sum = fmaf(x, x, sum);
  }
  const float scale = rsqrtf(ReduceSum(sum, shared) / width + epsilon);
  for (int i = threadIdx.x; i < width; i += 128)
    output[row * width + i] =
        input[row * width + i] * scale * (weight ? weight[i] : 1.0F);
}

__global__ void Rope(float* values, const float* factors, int width, int heads,
                     int position, float base) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= heads * width / 2)
    return;
  const int head = pair / (width / 2), i = pair % (width / 2);
  const float theta =
      position * powf(base, -2.0F * i / width) / (factors ? factors[i] : 1.0F);
  const float cosine = cosf(theta), sine = sinf(theta);
  const int first = head * width + i, second = first + width / 2;
  const float a = values[first], b = values[second];
  values[first] = a * cosine - b * sine;
  values[second] = a * sine + b * cosine;
}

__global__ void Attention(const float* query, const float* keys,
                          const float* values, float* output, int width,
                          int kv_heads, int query_heads, int capacity,
                          int position, int window, int last_position) {
  extern __shared__ float scores[];
  __shared__ float reduction[128];
  const int head = blockIdx.x, kv_head = head / (query_heads / kv_heads);
  const int first = window ? max(0, position - window + 1) : 0;
  const int count = last_position - first + 1;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int t = warp; t < count; t += 4) {
    const int slot = (first + t) % capacity;
    float dot = 0;
    for (int i = lane; i < width; i += 32)
      dot =
          fmaf(query[head * width + i],
               keys[(std::size_t(slot) * kv_heads + kv_head) * width + i], dot);
    for (int shift = 16; shift; shift >>= 1)
      dot += __shfl_down(dot, shift, 32);
    if (lane == 0)
      scores[t] = dot;
  }
  __syncthreads();
  float largest = -INFINITY;
  for (int t = threadIdx.x; t < count; t += 128)
    largest = fmaxf(largest, scores[t]);
  reduction[threadIdx.x] = largest;
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  largest = reduction[0];
  __syncthreads();
  float sum = 0;
  for (int t = threadIdx.x; t < count; t += 128) {
    const float value = expf(scores[t] - largest);
    scores[t] = value;
    sum += value;
  }
  const float inverse = 1.0F / ReduceSum(sum, reduction);
  for (int i = threadIdx.x; i < width; i += 128) {
    float result = 0;
    for (int t = 0; t < count; ++t) {
      const int slot = (first + t) % capacity;
      result = fmaf(
          scores[t] * inverse,
          values[(std::size_t(slot) * kv_heads + kv_head) * width + i], result);
    }
    output[head * width + i] = result;
  }
}

__global__ void GeluGate(float* gate, const float* up, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    const float x = gate[i];
    gate[i] =
        0.5F * x *
        (1.0F + tanhf(0.7978845608028654F * x * (1.0F + 0.044715F * x * x))) *
        up[i];
  }
}

__global__ void Residual(float* hidden, const float* update, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    hidden[i] += update[i];
}

__global__ void Scale(float* hidden, const float* scale, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    hidden[i] *= *scale;
}

__global__ void Softcap(float* logits, int size, float cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    logits[i] = cap * tanhf(logits[i] / cap);
}

struct Layer {
  const Tensor *pre, *q, *k, *v, *q_norm, *k_norm, *output, *post, *ffn_pre,
      *gate, *up, *down, *ffn_post, *scale;
};

struct BatchScratch {
  Buffer hidden{kMaxImageTokens * 5376 * sizeof(float)},
      norm{kMaxImageTokens * 5376 * sizeof(float)},
      q{kMaxImageTokens * 32 * 512 * sizeof(float)},
      k{kMaxImageTokens * 16 * 256 * sizeof(float)},
      v{kMaxImageTokens * 16 * 256 * sizeof(float)},
      attention{kMaxImageTokens * 32 * 512 * sizeof(float)},
      update{kMaxImageTokens * 5376 * sizeof(float)},
      gate{kMaxImageTokens * 21504 * sizeof(float)},
      up{kMaxImageTokens * 21504 * sizeof(float)},
      weights{std::size_t(5376) * 21504 * sizeof(float)};
  std::size_t Bytes() const {
    return hidden.bytes + norm.bytes + q.bytes + k.bytes + v.bytes +
           attention.bytes + update.bytes + gate.bytes + up.bytes +
           weights.bytes;
  }
};

struct Assistant {
  std::unordered_map<std::string, Tensor> weights;
  std::vector<Layer> layers;
  Buffer hidden{1024 * sizeof(float)}, norm{1024 * sizeof(float)},
      query{32 * 512 * sizeof(float)}, attention{32 * 512 * sizeof(float)},
      update{1024 * sizeof(float)}, gate{8192 * sizeof(float)},
      up{8192 * sizeof(float)}, concatenated{10752 * sizeof(float)},
      next_hidden{5376 * sizeof(float)}, logits{262144 * sizeof(float)};
  std::vector<float> host_logits = std::vector<float>(262144);
  const Tensor *pre{nullptr}, *post{nullptr}, *embedding{nullptr},
      *final_norm{nullptr}, *rope{nullptr};
  std::size_t ScratchBytes() const {
    return hidden.bytes + norm.bytes + query.bytes + attention.bytes +
           update.bytes + gate.bytes + up.bytes + concatenated.bytes +
           next_hidden.bytes + logits.bytes +
           host_logits.size() * sizeof(float);
  }
};

struct Cache {
  Buffer keys, values;
  std::uint32_t capacity{0}, width{0}, heads{0};
};

}  // namespace

void device::Normalize(hipStream_t stream, const float* input,
                       const float* weight, float* output, int width, int rows,
                       float epsilon) {
  hipLaunchKernelGGL(RmsNorm, dim3(rows), dim3(128), 0, stream, input, weight,
                     output, width, epsilon);
}
void device::Gate(hipStream_t stream, float* gate, const float* up, int size) {
  hipLaunchKernelGGL(GeluGate, dim3((size + 127) / 128), dim3(128), 0, stream,
                     gate, up, size);
}
void device::Add(hipStream_t stream, float* hidden, const float* update,
                 int size) {
  hipLaunchKernelGGL(Residual, dim3((size + 127) / 128), dim3(128), 0, stream,
                     hidden, update, size);
}

struct Model::Impl {
  std::unique_ptr<core::GgufReader> reader;
  Config config;
  std::unique_ptr<Tokenizer> tokenizer;
  std::uint32_t context{0};
  bool sessions_started{false};
  std::size_t weight_bytes{0};
  Stream stream;
  std::unordered_map<std::string, Tensor> weights;
  std::vector<Layer> layers;
  std::unique_ptr<Assistant> assistant;
  std::unique_ptr<Vision> vision;
  std::unique_ptr<BatchScratch> batch;
  std::unique_ptr<Blas> blas;
  Buffer hidden, norm, q, k, v, attention, update, gate, up;
  const Tensor *embedding{nullptr}, *final_norm{nullptr},
      *rope_factors{nullptr};

  std::size_t ScratchBytes() const {
    return hidden.bytes + norm.bytes + q.bytes + k.bytes + v.bytes +
           attention.bytes + update.bytes + gate.bytes + up.bytes +
           (assistant ? assistant->ScratchBytes() : 0) +
           (vision ? vision->ScratchBytes() : 0) + (batch ? batch->Bytes() : 0);
  }
};

struct Session::Impl {
  std::shared_ptr<const Model> model;
  Model::Impl* engine;
  std::vector<Cache> cache;
  std::size_t position{0};
  Buffer hidden, logits;
  std::vector<float> host_logits;
  std::string layer_directory, assistant_directory;
  std::function<bool()> cancelled;
  std::function<void(std::size_t, std::span<const float>)> logit_observer;

  explicit Impl(std::shared_ptr<const Model> owner)
      : model(std::move(owner)),
        engine(model->impl_.get()),
        hidden(engine->config.hidden * sizeof(float)),
        logits(engine->config.vocabulary * sizeof(float)),
        host_logits(engine->config.vocabulary) {
    const auto& c = engine->config;
    for (std::size_t i = 0; i < c.layers; ++i) {
      const auto capacity =
          c.Local(i) ? std::min<std::uint32_t>(
                           c.window + (engine->vision ? kMaxImageTokens
                                                      : Session::kPrefillBatch),
                           engine->context)
                     : engine->context;
      const std::size_t bytes =
          std::size_t(capacity) * c.HeadWidth(i) * c.KvHeads(i) * sizeof(float);
      cache.push_back({Buffer(bytes), Buffer(bytes), capacity, c.HeadWidth(i),
                       c.KvHeads(i)});
    }
  }

  void Capture(std::string_view name, const float* pointer,
               std::size_t count) const {
    if (layer_directory.empty())
      return;
    std::vector<float> values(count);
    Check(hipMemcpyAsync(values.data(), pointer, count * sizeof(float),
                         hipMemcpyDeviceToHost, engine->stream.value));
    Check(hipStreamSynchronize(engine->stream.value));
    std::filesystem::create_directories(layer_directory);
    std::ofstream output(
        std::filesystem::path(layer_directory) /
            (std::to_string(position) + "-" + std::string(name) + ".bin"),
        std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 count * sizeof(float));
    if (!output)
      throw std::runtime_error("Gemma 4 layer dump failed");
  }
};

struct Snapshot::Impl {
  std::shared_ptr<const Model> model;
  std::size_t position{0};
  std::vector<std::vector<float>> keys, values;
  std::vector<float> hidden, logits;
};

Snapshot::Snapshot(std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}
Snapshot::~Snapshot() = default;
std::size_t Snapshot::Position() const noexcept {
  return impl_->position;
}
std::span<const float> Snapshot::Logits() const noexcept {
  return impl_->logits;
}
std::string Snapshot::Digest() const {
  crypto::Sha256Hasher hash;
  const auto add = [&](const void* data, std::size_t bytes) {
    hash.Update({static_cast<const std::uint8_t*>(data), bytes});
  };
  add(&impl_->position, sizeof(impl_->position));
  for (std::size_t i = 0; i < impl_->keys.size(); ++i) {
    add(impl_->keys[i].data(), impl_->keys[i].size() * sizeof(float));
    add(impl_->values[i].data(), impl_->values[i].size() * sizeof(float));
  }
  add(impl_->hidden.data(), impl_->hidden.size() * sizeof(float));
  add(impl_->logits.data(), impl_->logits.size() * sizeof(float));
  return hash.FinishHex();
}
std::size_t Snapshot::PayloadBytes() const noexcept {
  std::size_t bytes =
      sizeof(impl_->position) +
      (impl_->hidden.size() + impl_->logits.size()) * sizeof(float);
  for (std::size_t i = 0; i < impl_->keys.size(); ++i)
    bytes += (impl_->keys[i].size() + impl_->values[i].size()) * sizeof(float);
  return bytes;
}

Model::Model(std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}
Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& path,
                                   std::uint32_t context) {
  if (context < 2 || context > 4096)
    throw std::invalid_argument(
        "Gemma 4 currently supports contexts from 2 to 4096");
  auto engine = std::make_unique<Impl>();
  std::string error;
  engine->reader = core::GgufReader::OpenFile(path, &error);
  if (!engine->reader)
    throw std::runtime_error(error);
  engine->config = Config::FromGguf(*engine->reader);
  engine->config.ValidateWeights(*engine->reader);
  engine->tokenizer = std::make_unique<Tokenizer>(*engine->reader);
  engine->context = context;
  for (const auto& tensor : engine->reader->GetTensors()) {
    Tensor device;
    device.columns = tensor.dimensions[0];
    device.rows = tensor.dimensions.size() > 1 ? tensor.dimensions[1] : 1;
    device.type = tensor.type;
    device.buffer = Buffer(tensor.size_bytes);
    Check(hipMemcpyAsync(device.buffer.data, tensor.data, tensor.size_bytes,
                         hipMemcpyHostToDevice, engine->stream.value));
    engine->weight_bytes += tensor.size_bytes;
    engine->weights.emplace(std::string(tensor.name), std::move(device));
  }
  Check(hipStreamSynchronize(engine->stream.value));
  const auto& c = engine->config;
  engine->hidden = Buffer(c.hidden * sizeof(float));
  engine->norm = Buffer(c.hidden * sizeof(float));
  engine->q = Buffer(c.heads * 512 * sizeof(float));
  engine->k = Buffer(16 * 256 * sizeof(float));
  engine->v = Buffer(16 * 256 * sizeof(float));
  engine->attention = Buffer(c.heads * 512 * sizeof(float));
  engine->update = Buffer(c.hidden * sizeof(float));
  engine->gate = Buffer(c.intermediate * sizeof(float));
  engine->up = Buffer(c.intermediate * sizeof(float));
  engine->batch = std::make_unique<BatchScratch>();
  engine->blas = std::make_unique<Blas>(engine->stream.value);
  engine->embedding = &engine->weights.at("token_embd.weight");
  engine->final_norm = &engine->weights.at("output_norm.weight");
  engine->rope_factors = &engine->weights.at("rope_freqs.weight");
  for (std::size_t i = 0; i < c.layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto get = [&](const char* name) {
      return &engine->weights.at(prefix + name + ".weight");
    };
    engine->layers.push_back(
        {get("attn_norm"), get("attn_q"), get("attn_k"),
         c.Local(i) ? get("attn_v") : nullptr, get("attn_q_norm"),
         get("attn_k_norm"), get("attn_output"), get("post_attention_norm"),
         get("ffn_norm"), get("ffn_gate"), get("ffn_up"), get("ffn_down"),
         get("post_ffw_norm"), get("layer_output_scale")});
  }
  return std::make_shared<Model>(std::move(engine));
}

std::unique_ptr<Session> Model::CreateSession() const {
  impl_->sessions_started = true;
  return std::make_unique<Session>(
      std::make_unique<Session::Impl>(shared_from_this()));
}
const Config& Model::config() const {
  return impl_->config;
}
const Tokenizer& Model::tokenizer() const {
  return *impl_->tokenizer;
}
std::string Model::Name() const {
  return std::string(
      impl_->reader->GetMetadataString("general.name").value_or("Gemma 4 31B"));
}
std::size_t Model::WeightBytes() const {
  return impl_->weight_bytes;
}
std::size_t Model::ScratchBytes() const {
  return impl_->ScratchBytes();
}
std::uint32_t Model::Context() const {
  return impl_->context;
}

bool Model::HasAssistant() const {
  return impl_->assistant != nullptr;
}

void Model::LoadAssistant(const std::string& path) {
  if (impl_->assistant)
    throw std::logic_error("Gemma assistant already loaded");
  std::string error;
  auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader)
    throw std::runtime_error(error);
  ValidateAssistant(*reader, *impl_->reader);
  const auto expected =
      impl_->embedding->type == core::GgmlType::kQ4_0
          ? "3a5e99fd8d0b23afb1fccd1ee0c9ebd1f571d00399c2dae2292d217feeec0f6b"
          : "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de";
  if (crypto::Sha256FileHex(path) != expected)
    throw std::invalid_argument(
        "Gemma assistant requires the matching validated sidecar artifact");
  auto assistant = std::make_unique<Assistant>();
  std::size_t bytes = 0;
  for (const auto& tensor : reader->GetTensors()) {
    Tensor device;
    device.columns = tensor.dimensions[0];
    device.rows = tensor.dimensions.size() > 1 ? tensor.dimensions[1] : 1;
    device.type = tensor.type;
    device.buffer = Buffer(tensor.size_bytes);
    Check(hipMemcpyAsync(device.buffer.data, tensor.data, tensor.size_bytes,
                         hipMemcpyHostToDevice, impl_->stream.value));
    bytes += tensor.size_bytes;
    assistant->weights.emplace(std::string(tensor.name), std::move(device));
  }
  Check(hipStreamSynchronize(impl_->stream.value));
  assistant->pre = &assistant->weights.at("nextn.pre_projection.weight");
  assistant->post = &assistant->weights.at("nextn.post_projection.weight");
  assistant->embedding = &assistant->weights.at("token_embd.weight");
  assistant->final_norm = &assistant->weights.at("output_norm.weight");
  assistant->rope = &assistant->weights.at("rope_freqs.weight");
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto get = [&](const char* name) {
      return &assistant->weights.at(prefix + name + ".weight");
    };
    assistant->layers.push_back(
        {get("attn_norm"), get("attn_q"), nullptr, nullptr, get("attn_q_norm"),
         nullptr, get("attn_output"), get("post_attention_norm"),
         get("ffn_norm"), get("ffn_gate"), get("ffn_up"), get("ffn_down"),
         get("post_ffw_norm"), get("layer_output_scale")});
  }
  impl_->weight_bytes += bytes;
  impl_->assistant = std::move(assistant);
}

void Model::LoadVision(const std::string& path) {
  if (impl_->vision)
    throw std::logic_error("Gemma vision already loaded");
  if (impl_->sessions_started)
    throw std::logic_error("load Gemma vision before creating sessions");
  auto vision = Vision::Load(path);
  const bool qat = impl_->embedding->type == core::GgmlType::kQ4_0;
  const std::string expected =
      qat ? "d904b3579a9fbfbd50bc9bf40cb7384909edbd69fa9276db5ddc853e80f0edca"
          : "7a4601b12ec680f706a7e0c7e1f78f579b1db64d485a9e352ee87d4b9daa45e4";
  if (vision->Identity() != expected)
    throw std::invalid_argument(
        "Gemma vision requires the matching validated BF16 projector artifact");
  impl_->weight_bytes += vision->WeightBytes();
  impl_->vision = std::move(vision);
}
bool Model::HasVision() const {
  return impl_->vision != nullptr;
}
const std::string& Model::VisionIdentity() const {
  if (!impl_->vision)
    throw std::logic_error("Gemma vision not loaded");
  return impl_->vision->Identity();
}
EncodedImage Model::EncodeImage(std::span<const std::uint8_t> bytes,
                                const std::string& directory) {
  if (!impl_->vision)
    throw std::invalid_argument(
        "Gemma image input requires its compatible BF16 projector");
  return impl_->vision->Encode(core::DecodeImage(bytes), {}, directory);
}
EncodedImage Model::EncodePixels(const core::Image& pixels,
                                 const std::function<bool()>& cancelled) {
  if (!impl_->vision)
    throw std::invalid_argument(
        "Gemma image input requires its compatible BF16 projector");
  return impl_->vision->Encode(pixels, cancelled);
}
std::size_t Model::KvBytes() const {
  const auto& c = impl_->config;
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < c.layers; ++i)
    bytes +=
        std::size_t(c.HeadWidth(i)) * c.KvHeads(i) * 2 * sizeof(float) *
        (c.Local(i) ? std::min<std::uint32_t>(
                          c.window + (HasVision() ? kMaxImageTokens
                                                  : Session::kPrefillBatch),
                          impl_->context)
                    : impl_->context);
  return bytes;
}

Session::Session(std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}
Session::~Session() = default;
std::size_t Session::Position() const {
  return impl_->position;
}
std::span<const float> Session::Logits() const {
  if (!impl_->position)
    throw std::logic_error("Gemma 4 has no evaluated frontier");
  return impl_->host_logits;
}
void Session::Reset() noexcept {
  impl_->position = 0;
}
void Session::SetLayerDirectory(std::string directory) {
  impl_->layer_directory = std::move(directory);
}
void Session::SetCancellationCheck(std::function<bool()> check) {
  impl_->cancelled = std::move(check);
}

std::size_t Session::SnapshotPayloadBytes() const {
  if (!impl_->position)
    throw std::logic_error("Gemma 4 has no snapshot frontier");
  std::size_t bytes =
      sizeof(impl_->position) + impl_->hidden.bytes + impl_->logits.bytes;
  for (const auto& cache : impl_->cache)
    bytes += 2 * std::min<std::size_t>(impl_->position, cache.capacity) *
             cache.width * cache.heads * sizeof(float);
  return bytes;
}

std::unique_ptr<Snapshot> Session::Save() const {
  (void)SnapshotPayloadBytes();
  auto result = std::make_unique<Snapshot::Impl>();
  result->model = impl_->model;
  result->position = impl_->position;
  result->logits = impl_->host_logits;
  result->hidden.resize(impl_->engine->config.hidden);
  // Capture a frozen state on a private stream, without touching model scratch.
  Stream capture;
  Check(hipMemcpyAsync(result->hidden.data(), impl_->hidden.data,
                       impl_->hidden.bytes, hipMemcpyDeviceToHost,
                       capture.value));
  for (const auto& cache : impl_->cache) {
    const std::size_t count =
        std::min<std::size_t>(result->position, cache.capacity) * cache.width *
        cache.heads;
    result->keys.emplace_back(count);
    result->values.emplace_back(count);
    Check(hipMemcpyAsync(result->keys.back().data(), cache.keys.data,
                         count * sizeof(float), hipMemcpyDeviceToHost,
                         capture.value));
    Check(hipMemcpyAsync(result->values.back().data(), cache.values.data,
                         count * sizeof(float), hipMemcpyDeviceToHost,
                         capture.value));
  }
  Check(hipStreamSynchronize(capture.value));
  return std::unique_ptr<Snapshot>(new Snapshot(std::move(result)));
}

void Session::Restore(const Snapshot& snapshot) {
  const auto& source = *snapshot.impl_;
  if (source.model.get() != impl_->model.get() || !source.position ||
      source.position > impl_->engine->context)
    throw std::invalid_argument(
        "Gemma 4 snapshot belongs to another model/context");
  const auto stream = impl_->engine->stream.value;
  try {
    for (std::size_t i = 0; i < impl_->cache.size(); ++i) {
      const auto& cache = impl_->cache[i];
      const std::size_t count =
          std::min<std::size_t>(source.position, cache.capacity) * cache.width *
          cache.heads;
      Check(hipMemcpyAsync(cache.keys.data, source.keys[i].data(),
                           count * sizeof(float), hipMemcpyHostToDevice,
                           stream));
      Check(hipMemcpyAsync(cache.values.data, source.values[i].data(),
                           count * sizeof(float), hipMemcpyHostToDevice,
                           stream));
    }
    Check(hipMemcpyAsync(impl_->hidden.data, source.hidden.data(),
                         impl_->hidden.bytes, hipMemcpyHostToDevice, stream));
    Check(hipMemcpyAsync(impl_->logits.data, source.logits.data(),
                         impl_->logits.bytes, hipMemcpyHostToDevice, stream));
    Check(hipStreamSynchronize(stream));
    impl_->host_logits = source.logits;
    impl_->position = source.position;
  } catch (...) {
    (void)hipStreamSynchronize(stream);
    Reset();
    throw;
  }
}

void Session::SetAssistantDirectory(std::string directory) {
  impl_->assistant_directory = std::move(directory);
}

std::vector<std::uint32_t> Session::Draft(std::uint32_t seed,
                                          std::size_t count) {
  auto& state = *impl_;
  auto& engine = *state.engine;
  if (!engine.assistant || !state.position ||
      seed >= engine.config.vocabulary || count > 7)
    throw std::invalid_argument("invalid Gemma assistant draft frontier");
  auto& a = *engine.assistant;
  const auto stream = engine.stream.value;
  std::vector<std::uint32_t> proposals;
  Check(hipMemcpyAsync(a.next_hidden.data, state.hidden.data,
                       state.hidden.bytes, hipMemcpyDeviceToDevice, stream));
  const auto capture = [&](std::string name, const float* data,
                           std::size_t size) {
    if (state.assistant_directory.empty())
      return;
    std::vector<float> host(size);
    Check(hipMemcpyAsync(host.data(), data, size * sizeof(float),
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    std::filesystem::create_directories(state.assistant_directory);
    std::ofstream output(
        std::filesystem::path(state.assistant_directory) /
            (std::to_string(proposals.size()) + "-" + name + ".bin"),
        std::ios::binary);
    output.write(reinterpret_cast<const char*>(host.data()),
                 size * sizeof(float));
    if (!output)
      throw std::runtime_error("Gemma assistant layer dump failed");
  };
  for (std::size_t step = 0; step < count; ++step) {
    if (state.cancelled && state.cancelled())
      throw std::runtime_error("Gemma assistant cancelled");
    Embed(*engine.embedding, seed, a.concatenated.floats(), std::sqrt(5376.0F),
          stream);
    Check(hipMemcpyAsync(a.concatenated.floats() + 5376, a.next_hidden.data,
                         state.hidden.bytes, hipMemcpyDeviceToDevice, stream));
    MatVec(*a.pre, a.concatenated.floats(), a.hidden.floats(), stream);
    capture("pre_proj", a.hidden.floats(), 1024);
    for (std::size_t i = 0; i < 4; ++i) {
      if (state.cancelled && state.cancelled())
        throw std::runtime_error("Gemma assistant cancelled");
      const auto& layer = a.layers[i];
      const bool local = i != 3;
      const auto& cache = state.cache[local ? 58 : 59];
      const auto width = cache.width;
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         a.hidden.floats(), layer.pre->buffer.floats(),
                         a.norm.floats(), 1024, 1e-6F);
      MatVec(*layer.q, a.norm.floats(), a.query.floats(), stream);
      hipLaunchKernelGGL(RmsNorm, dim3(32), dim3(128), 0, stream,
                         a.query.floats(), layer.q_norm->buffer.floats(),
                         a.query.floats(), width, 1e-6F);
      hipLaunchKernelGGL(Rope, dim3((32 * width / 2 + 127) / 128), dim3(128), 0,
                         stream, a.query.floats(),
                         local ? nullptr : a.rope->buffer.floats(), width, 32,
                         state.position, local ? 10000.0F : 1000000.0F);
      const auto visible =
          local ? std::min<std::size_t>(1023, state.position) : state.position;
      hipLaunchKernelGGL(
          Attention, dim3(32), dim3(128), visible * sizeof(float), stream,
          a.query.floats(), cache.keys.floats(), cache.values.floats(),
          a.attention.floats(), width, cache.heads, 32, cache.capacity,
          state.position, local ? 1024 : 0, state.position - 1);
      MatVec(*layer.output, a.attention.floats(), a.update.floats(), stream);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         a.update.floats(), layer.post->buffer.floats(),
                         a.update.floats(), 1024, 1e-6F);
      hipLaunchKernelGGL(Residual, dim3(8), dim3(128), 0, stream,
                         a.hidden.floats(), a.update.floats(), 1024);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         a.hidden.floats(), layer.ffn_pre->buffer.floats(),
                         a.norm.floats(), 1024, 1e-6F);
      MatVec(*layer.gate, a.norm.floats(), a.gate.floats(), stream);
      MatVec(*layer.up, a.norm.floats(), a.up.floats(), stream);
      hipLaunchKernelGGL(GeluGate, dim3(64), dim3(128), 0, stream,
                         a.gate.floats(), a.up.floats(), 8192);
      MatVec(*layer.down, a.gate.floats(), a.update.floats(), stream);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         a.update.floats(), layer.ffn_post->buffer.floats(),
                         a.update.floats(), 1024, 1e-6F);
      hipLaunchKernelGGL(Residual, dim3(8), dim3(128), 0, stream,
                         a.hidden.floats(), a.update.floats(), 1024);
      hipLaunchKernelGGL(Scale, dim3(8), dim3(128), 0, stream,
                         a.hidden.floats(), layer.scale->buffer.floats(), 1024);
      capture("l_out-" + std::to_string(i), a.hidden.floats(), 1024);
    }
    hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                       a.hidden.floats(), a.final_norm->buffer.floats(),
                       a.norm.floats(), 1024, 1e-6F);
    MatVec(*a.embedding, a.norm.floats(), a.logits.floats(), stream);
    MatVec(*a.post, a.norm.floats(), a.next_hidden.floats(), stream);
    capture("result_output", a.logits.floats(), 262144);
    capture("h_nextn", a.next_hidden.floats(), 5376);
    Check(hipGetLastError());
    Check(hipMemcpyAsync(a.host_logits.data(), a.logits.data, a.logits.bytes,
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    if (!std::ranges::all_of(a.host_logits,
                             [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("non-finite Gemma assistant logits");
    seed = std::max_element(a.host_logits.begin(), a.host_logits.end()) -
           a.host_logits.begin();
    proposals.push_back(seed);
    if (Tokenizer::Stop(seed))
      break;
  }
  return proposals;
}

void Session::EvaluateImage(std::span<const float> embeddings) {
  auto& state = *impl_;
  auto& e = *state.engine;
  const auto& c = e.config;
  if (!e.vision || embeddings.empty() || embeddings.size() % c.hidden ||
      embeddings.size() / c.hidden > kMaxImageTokens ||
      !std::ranges::all_of(embeddings,
                           [](float x) { return std::isfinite(x); }))
    throw std::invalid_argument("invalid Gemma image embeddings");
  EvaluateBatch({}, embeddings);
}

void Session::EvaluateTokens(std::span<const std::uint32_t> tokens) {
  if (tokens.empty() || tokens.size() > kPrefillBatch ||
      !std::ranges::all_of(tokens, [&](auto id) {
        return id < impl_->engine->config.vocabulary;
      }))
    throw std::invalid_argument("invalid Gemma text prefill batch");
  if (tokens.size() == 1) {
    Evaluate(tokens.front());
    return;
  }
  EvaluateBatch(tokens, {});
}

void Session::SetLogitObserver(
    std::function<void(std::size_t, std::span<const float>)> observer) {
  impl_->logit_observer = std::move(observer);
}

void Session::EvaluateBatch(std::span<const std::uint32_t> tokens,
                            std::span<const float> embeddings) {
  auto& state = *impl_;
  auto& e = *state.engine;
  const auto& c = e.config;
  const bool image = tokens.empty();
  const std::size_t count =
      image ? embeddings.size() / c.hidden : tokens.size();
  if (count > e.context - state.position)
    throw std::out_of_range("Gemma prefill exceeds context");
  const auto stream = e.stream.value;
  auto& b = *e.batch;
  try {
    if (image) {
      Check(hipMemcpyAsync(b.hidden.data, embeddings.data(),
                           embeddings.size_bytes(), hipMemcpyHostToDevice,
                           stream));
    } else {
      for (std::size_t t = 0; t < count; ++t)
        Embed(*e.embedding, tokens[t], b.hidden.floats() + t * c.hidden,
              std::sqrt(float(c.hidden)), stream);
    }
    for (std::size_t i = 0; i < c.layers; ++i) {
      if (state.cancelled && state.cancelled())
        throw std::runtime_error("Gemma prefill cancelled");
      const auto& layer = e.layers[i];
      auto& cache = state.cache[i];
      const auto width = c.HeadWidth(i), kv_heads = c.KvHeads(i);
      const std::size_t query_row = c.heads * width, kv_row = kv_heads * width;
      const float* factors =
          c.Local(i) ? nullptr : e.rope_factors->buffer.floats();
      const float base = c.Local(i) ? c.local_rope_base : c.global_rope_base;
      Normalize(stream, b.hidden.floats(), layer.pre->buffer.floats(),
                b.norm.floats(), c.hidden, count);
      MatGemm(*layer.q, b.norm.floats(), b.q.floats(), count, *e.blas,
              b.weights, stream);
      MatGemm(*layer.k, b.norm.floats(), b.k.floats(), count, *e.blas,
              b.weights, stream);
      if (layer.v)
        MatGemm(*layer.v, b.norm.floats(), b.v.floats(), count, *e.blas,
                b.weights, stream);
      else
        Check(hipMemcpyAsync(b.v.data, b.k.data, count * kv_row * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
      Normalize(stream, b.q.floats(), layer.q_norm->buffer.floats(),
                b.q.floats(), width, c.heads * count);
      Normalize(stream, b.k.floats(), layer.k_norm->buffer.floats(),
                b.k.floats(), width, kv_heads * count);
      Normalize(stream, b.v.floats(), nullptr, b.v.floats(), width,
                kv_heads * count);
      for (std::size_t t = 0; t < count; ++t) {
        const auto pos = state.position + t;
        hipLaunchKernelGGL(Rope, dim3((query_row / 2 + 127) / 128), dim3(128),
                           0, stream, b.q.floats() + t * query_row, factors,
                           width, c.heads, pos, base);
        hipLaunchKernelGGL(Rope, dim3((kv_row / 2 + 127) / 128), dim3(128), 0,
                           stream, b.k.floats() + t * kv_row, factors, width,
                           kv_heads, pos, base);
        const auto offset = (pos % cache.capacity) * kv_row;
        Check(hipMemcpyAsync(cache.keys.floats() + offset,
                             b.k.floats() + t * kv_row, kv_row * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
        Check(hipMemcpyAsync(cache.values.floats() + offset,
                             b.v.floats() + t * kv_row, kv_row * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
      }
      for (std::size_t t = 0; t < count; ++t) {
        const auto pos = state.position + t;
        // Only local image layers see future rows. Extra ring capacity keeps
        // each query's past intact while writing the complete batch.
        const auto last =
            image && c.Local(i) ? state.position + count - 1 : pos;
        const auto first =
            c.Local(i) && pos >= c.window ? pos - c.window + 1 : 0;
        hipLaunchKernelGGL(Attention, dim3(c.heads), dim3(128),
                           (last - first + 1) * sizeof(float), stream,
                           b.q.floats() + t * query_row, cache.keys.floats(),
                           cache.values.floats(),
                           b.attention.floats() + t * query_row, width,
                           kv_heads, c.heads, cache.capacity, pos,
                           c.Local(i) ? c.window : 0, last);
      }
      MatGemm(*layer.output, b.attention.floats(), b.update.floats(), count,
              *e.blas, b.weights, stream);
      Normalize(stream, b.update.floats(), layer.post->buffer.floats(),
                b.update.floats(), c.hidden, count);
      Add(stream, b.hidden.floats(), b.update.floats(), count * c.hidden);
      Normalize(stream, b.hidden.floats(), layer.ffn_pre->buffer.floats(),
                b.norm.floats(), c.hidden, count);
      MatGemm(*layer.gate, b.norm.floats(), b.gate.floats(), count, *e.blas,
              b.weights, stream);
      MatGemm(*layer.up, b.norm.floats(), b.up.floats(), count, *e.blas,
              b.weights, stream);
      Gate(stream, b.gate.floats(), b.up.floats(), count * c.intermediate);
      MatGemm(*layer.down, b.gate.floats(), b.update.floats(), count, *e.blas,
              b.weights, stream);
      Normalize(stream, b.update.floats(), layer.ffn_post->buffer.floats(),
                b.update.floats(), c.hidden, count);
      Add(stream, b.hidden.floats(), b.update.floats(), count * c.hidden);
      hipLaunchKernelGGL(Scale, dim3((count * c.hidden + 127) / 128), dim3(128),
                         0, stream, b.hidden.floats(),
                         layer.scale->buffer.floats(), count * c.hidden);
    }
    // Production needs only the final frontier. Numerical diagnostics can
    // observe every row without changing the transformer or its KV writes.
    for (std::size_t t = state.logit_observer ? 0 : count - 1; t < count; ++t) {
      Normalize(stream, b.hidden.floats() + t * c.hidden,
                e.final_norm->buffer.floats(), state.hidden.floats(), c.hidden,
                1);
      MatVec(*e.embedding, state.hidden.floats(), state.logits.floats(),
             stream);
      hipLaunchKernelGGL(Softcap, dim3((c.vocabulary + 127) / 128), dim3(128),
                         0, stream, state.logits.floats(), c.vocabulary,
                         c.softcap);
      Check(hipGetLastError());
      Check(hipMemcpyAsync(state.host_logits.data(), state.logits.data,
                           state.logits.bytes, hipMemcpyDeviceToHost, stream));
      Check(hipStreamSynchronize(stream));
      if (!std::ranges::all_of(state.host_logits,
                               [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("non-finite Gemma prefill logits");
      if (state.logit_observer)
        state.logit_observer(state.position + t + 1, state.host_logits);
    }
    state.position += count;
  } catch (...) {
    (void)hipStreamSynchronize(stream);
    Reset();
    throw;
  }
}

void Session::Evaluate(std::uint32_t token) {
  auto& state = *impl_;
  auto& e = *state.engine;
  const auto& c = e.config;
  if (token >= c.vocabulary)
    throw std::invalid_argument("Gemma 4 token outside vocabulary");
  if (state.position >= e.context)
    throw std::out_of_range("Gemma 4 context exhausted");
  const auto stream = e.stream.value;
  try {
    Embed(*e.embedding, token, e.hidden.floats(), std::sqrt(float(c.hidden)),
          stream);
    state.Capture("inp_scaled", e.hidden.floats(), c.hidden);
    for (std::size_t i = 0; i < c.layers; ++i) {
      if (state.cancelled && state.cancelled())
        throw std::runtime_error("Gemma 4 evaluation cancelled");
      const auto& layer = e.layers[i];
      auto& cache = state.cache[i];
      const auto width = c.HeadWidth(i), kv_heads = c.KvHeads(i);
      const float* factors =
          c.Local(i) ? nullptr : e.rope_factors->buffer.floats();
      const float base = c.Local(i) ? c.local_rope_base : c.global_rope_base;
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         e.hidden.floats(), layer.pre->buffer.floats(),
                         e.norm.floats(), c.hidden, c.epsilon);
      if (i == 0)
        state.Capture("attn_norm-0", e.norm.floats(), c.hidden);
      MatVec(*layer.q, e.norm.floats(), e.q.floats(), stream);
      MatVec(*layer.k, e.norm.floats(), e.k.floats(), stream);
      if (layer.v)
        MatVec(*layer.v, e.norm.floats(), e.v.floats(), stream);
      else
        Check(hipMemcpyAsync(e.v.data, e.k.data,
                             width * kv_heads * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
      hipLaunchKernelGGL(RmsNorm, dim3(c.heads), dim3(128), 0, stream,
                         e.q.floats(), layer.q_norm->buffer.floats(),
                         e.q.floats(), width, c.epsilon);
      hipLaunchKernelGGL(RmsNorm, dim3(kv_heads), dim3(128), 0, stream,
                         e.k.floats(), layer.k_norm->buffer.floats(),
                         e.k.floats(), width, c.epsilon);
      hipLaunchKernelGGL(RmsNorm, dim3(kv_heads), dim3(128), 0, stream,
                         e.v.floats(), nullptr, e.v.floats(), width, c.epsilon);
      hipLaunchKernelGGL(Rope, dim3((c.heads * width / 2 + 127) / 128),
                         dim3(128), 0, stream, e.q.floats(), factors, width,
                         c.heads, state.position, base);
      hipLaunchKernelGGL(Rope, dim3((kv_heads * width / 2 + 127) / 128),
                         dim3(128), 0, stream, e.k.floats(), factors, width,
                         kv_heads, state.position, base);
      if (i == 0) {
        state.Capture("Qcur_pos-0", e.q.floats(), c.heads * width);
        state.Capture("Kcur_pos-0", e.k.floats(), kv_heads * width);
        state.Capture("Vcur_normed-0", e.v.floats(), kv_heads * width);
      }
      const std::size_t row = std::size_t(width) * kv_heads;
      const auto offset = (state.position % cache.capacity) * row;
      Check(hipMemcpyAsync(cache.keys.floats() + offset, e.k.data,
                           row * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
      Check(hipMemcpyAsync(cache.values.floats() + offset, e.v.data,
                           row * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
      const int window = c.Local(i) ? c.window : 0;
      const auto visible =
          c.Local(i) ? std::min<std::size_t>(c.window, state.position + 1)
                     : state.position + 1;
      hipLaunchKernelGGL(
          Attention, dim3(c.heads), dim3(128), visible * sizeof(float), stream,
          e.q.floats(), cache.keys.floats(), cache.values.floats(),
          e.attention.floats(), width, kv_heads, c.heads, cache.capacity,
          state.position, window, state.position);
      MatVec(*layer.output, e.attention.floats(), e.update.floats(), stream);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         e.update.floats(), layer.post->buffer.floats(),
                         e.update.floats(), c.hidden, c.epsilon);
      hipLaunchKernelGGL(Residual, dim3((c.hidden + 127) / 128), dim3(128), 0,
                         stream, e.hidden.floats(), e.update.floats(),
                         c.hidden);
      if (i == 0)
        state.Capture("attn_out-0", e.hidden.floats(), c.hidden);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         e.hidden.floats(), layer.ffn_pre->buffer.floats(),
                         e.norm.floats(), c.hidden, c.epsilon);
      MatVec(*layer.gate, e.norm.floats(), e.gate.floats(), stream);
      MatVec(*layer.up, e.norm.floats(), e.up.floats(), stream);
      hipLaunchKernelGGL(GeluGate, dim3((c.intermediate + 127) / 128),
                         dim3(128), 0, stream, e.gate.floats(), e.up.floats(),
                         c.intermediate);
      MatVec(*layer.down, e.gate.floats(), e.update.floats(), stream);
      hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                         e.update.floats(), layer.ffn_post->buffer.floats(),
                         e.update.floats(), c.hidden, c.epsilon);
      hipLaunchKernelGGL(Residual, dim3((c.hidden + 127) / 128), dim3(128), 0,
                         stream, e.hidden.floats(), e.update.floats(),
                         c.hidden);
      hipLaunchKernelGGL(Scale, dim3((c.hidden + 127) / 128), dim3(128), 0,
                         stream, e.hidden.floats(),
                         layer.scale->buffer.floats(), c.hidden);
      if (i == 0 || i == 5)
        state.Capture("l_out-" + std::to_string(i), e.hidden.floats(),
                      c.hidden);
    }
    hipLaunchKernelGGL(RmsNorm, dim3(1), dim3(128), 0, stream,
                       e.hidden.floats(), e.final_norm->buffer.floats(),
                       state.hidden.floats(), c.hidden, c.epsilon);
    state.Capture("result_norm", state.hidden.floats(), c.hidden);
    MatVec(*e.embedding, state.hidden.floats(), state.logits.floats(), stream);
    hipLaunchKernelGGL(Softcap, dim3((c.vocabulary + 127) / 128), dim3(128), 0,
                       stream, state.logits.floats(), c.vocabulary, c.softcap);
    Check(hipGetLastError());
    Check(hipMemcpyAsync(state.host_logits.data(), state.logits.data,
                         state.logits.bytes, hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    if (!std::ranges::all_of(state.host_logits,
                             [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("non-finite Gemma 4 logits");
    ++state.position;
    if (state.logit_observer)
      state.logit_observer(state.position, state.host_logits);
  } catch (...) {
    (void)hipStreamSynchronize(stream);
    Reset();
    throw;
  }
}

}  // namespace gufo::models::gemma4
