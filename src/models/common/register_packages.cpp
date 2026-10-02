#include "src/models/common/register_packages.hpp"

#include <mutex>

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/deepseek_v4_flash/serve_runner.hpp"
#include "src/models/qwen/serve_runner.hpp"
#include "src/models/qwen38_flash_next/serve_runner.hpp"
#endif

namespace gufo::models::common {

void RegisterAllModelPackages() {
#if defined(ENGINE_ENABLE_HIP)
  static std::once_flag registered;
  std::call_once(registered, [] {
    qwen::serve::RegisterQwenPackage();
    deepseek_v4_flash::serve::RegisterDeepSeekPackage();
    qwen38_flash_next::serve::RegisterFlashNextPackage();
  });
#else
  // No serving-capable package without the HIP backend; file loads fail with
  // the same "requires the HIP backend" error as before.
#endif
}

}  // namespace gufo::models::common
