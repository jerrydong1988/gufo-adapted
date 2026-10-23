// WINDOWS: Reuse the upstream upload-lifetime fixture without ELF --wrap.
// This executable owns its HIP import pointer; production targets are
// untouched.
#include <hip/hip_runtime.h>
#include <windows.h>

#include <cstdlib>

using Copy = hipError_t (*)(void*, const void*, std::size_t, hipMemcpyKind,
                            hipStream_t);

extern "C" hipError_t __wrap_hipMemcpyAsync(void*, const void*, std::size_t,
                                            hipMemcpyKind, hipStream_t);

extern "C" {
Copy __imp_hipMemcpyAsync = __wrap_hipMemcpyAsync;

hipError_t __real_hipMemcpyAsync(void* destination, const void* source,
                                 std::size_t bytes, hipMemcpyKind kind,
                                 hipStream_t stream) {
  static const auto real = [] {
    const auto module = GetModuleHandleW(L"amdhip64_7.dll");
    if (!module)
      std::abort();
    const auto symbol = GetProcAddress(module, "hipMemcpyAsync");
    if (!symbol)
      std::abort();
    return reinterpret_cast<Copy>(symbol);
  }();
  return real(destination, source, bytes, kind, stream);
}
}
