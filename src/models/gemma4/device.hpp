#ifndef GUFO_MODELS_GEMMA4_DEVICE_HPP_
#define GUFO_MODELS_GEMMA4_DEVICE_HPP_

// Model-private allocation and FP32 operations. No cross-model GPU contract.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::device {
inline void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("Gemma 4 HIP: ") +
                             hipGetErrorString(status));
}
inline void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Gemma 4 hipBLAS failed: " +
                             std::to_string(status));
}
struct Buffer {
  void* data{nullptr};
  std::size_t bytes{0};
  Buffer() = default;
  explicit Buffer(std::size_t size) : bytes(size) {
    Check(hipMalloc(&data, size));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : data(std::exchange(other.data, nullptr)),
        bytes(std::exchange(other.bytes, 0)) {}
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (data)
        (void)hipFree(data);
      data = std::exchange(other.data, nullptr);
      bytes = std::exchange(other.bytes, 0);
    }
    return *this;
  }
  float* floats() const { return static_cast<float*>(data); }
};
struct Stream {
  hipStream_t value{};
  Stream() { Check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() {
    if (value)
      (void)hipStreamDestroy(value);
  }
};
struct Tensor {
  Buffer buffer;
  int columns{0}, rows{0};
  core::GgmlType type{core::GgmlType::kF32};
};
struct Blas {
  hipblasHandle_t handle{};
  explicit Blas(hipStream_t stream) {
    CheckBlas(hipblasCreate(&handle));
    CheckBlas(hipblasSetStream(handle, stream));
    CheckBlas(hipblasSetAtomicsMode(handle, HIPBLAS_ATOMICS_NOT_ALLOWED));
  }
  ~Blas() {
    if (handle)
      (void)hipblasDestroy(handle);
  }
  void Multiply(const float* weights, const float* input, float* output,
                int columns, int rows, int batch) const {
    const float alpha = 1, beta = 0;
    CheckBlas(hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, rows, batch,
                           columns, &alpha, weights, columns, input, columns,
                           &beta, output, rows));
  }
};
void Normalize(hipStream_t stream, const float* input, const float* weight,
               float* output, int width, int rows, float epsilon = 1e-6F);
void Gate(hipStream_t stream, float* gate, const float* up, int size);
void Add(hipStream_t stream, float* hidden, const float* update, int size);
}  // namespace gufo::models::gemma4::device
#endif
