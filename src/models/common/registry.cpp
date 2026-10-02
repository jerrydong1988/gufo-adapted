#include "src/models/common/registry.hpp"

#include "src/core/gguf_reader.hpp"

namespace gufo::models::common {

const TextModelPackage* TextModelRegistry::FindForReader(
    const core::GgufReader& reader) const {
  const std::string architecture(
      reader.GetMetadataString("general.architecture").value_or(""));
  return FindForArchitecture(architecture);
}

}  // namespace gufo::models::common
