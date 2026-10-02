#ifndef GUFO_MODELS_COMMON_REGISTER_PACKAGES_HPP_
#define GUFO_MODELS_COMMON_REGISTER_PACKAGES_HPP_

// Registers every compiled-in text model package with the global registry.
// Serve, prompt, and bench call this once before resolving a package. Without
// the HIP backend no serving-capable package registers, matching today's
// behavior where inference requires a GPU build.

namespace gufo::models::common {

void RegisterAllModelPackages();

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_REGISTER_PACKAGES_HPP_
