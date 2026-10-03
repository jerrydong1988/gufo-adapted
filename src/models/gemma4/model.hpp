#ifndef GUFO_MODELS_GEMMA4_MODEL_HPP_
#define GUFO_MODELS_GEMMA4_MODEL_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/common/chat.hpp"

namespace gufo::core {
struct Image;
}

namespace gufo::models::gemma4 {

struct Config {
  std::uint32_t layers{60};
  std::uint32_t hidden{5376};
  std::uint32_t intermediate{21504};
  std::uint32_t heads{32};
  std::uint32_t vocabulary{262144};
  std::uint32_t window{1024};
  std::uint32_t native_context{262144};
  float epsilon{1e-6F};
  float softcap{30.0F};
  float local_rope_base{10000.0F};
  float global_rope_base{1000000.0F};
  [[nodiscard]] bool Local(std::size_t layer) const { return layer % 6 != 5; }
  [[nodiscard]] std::uint32_t HeadWidth(std::size_t layer) const {
    return Local(layer) ? 256 : 512;
  }
  [[nodiscard]] std::uint32_t KvHeads(std::size_t layer) const {
    return Local(layer) ? 16 : 4;
  }
  [[nodiscard]] static Config FromGguf(const core::GgufReader& reader);
  void ValidateWeights(const core::GgufReader& reader) const;
  [[nodiscard]] std::size_t KvBytes(std::uint32_t context) const;
};

class Tokenizer {
public:
  explicit Tokenizer(const core::GgufReader& reader);
  [[nodiscard]] std::vector<std::uint32_t> Encode(
      std::string_view text, bool add_special = true) const;
  [[nodiscard]] std::string Decode(std::span<const std::uint32_t> ids) const;
  [[nodiscard]] static bool Stop(std::uint32_t id) {
    return id == 1 || id == 106 || id == 50;
  }

private:
  void EncodePlain(std::string_view text,
                   std::vector<std::uint32_t>& ids) const;
  std::vector<std::string> tokens_;
  std::unordered_map<std::string, std::uint32_t> token_ids_;
  std::unordered_map<std::string, std::uint32_t> merges_;
  std::vector<std::pair<std::string, std::uint32_t>> special_;
  std::array<std::uint32_t, 256> bytes_{};
  bool add_bos_{false};
};

[[nodiscard]] bool ValidateTemplate(const core::GgufReader& reader,
                                    std::string* error);
void ValidateAssistant(const core::GgufReader& assistant,
                       const core::GgufReader& target);
struct RenderedImage {
  std::size_t offset;
  std::shared_ptr<const std::vector<std::uint8_t>> bytes;
};
[[nodiscard]] std::string RenderChat(
    std::span<const common::ChatMessage> messages,
    const ReasoningOptions& reasoning,
    std::vector<RenderedImage>* images = nullptr);

class Session;
struct EncodedImage;
class Snapshot {
public:
  struct Impl;
  ~Snapshot();
  [[nodiscard]] std::size_t PayloadBytes() const noexcept;
  [[nodiscard]] std::size_t Position() const noexcept;
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::string Digest() const;

private:
  explicit Snapshot(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

class Model : public std::enable_shared_from_this<Model> {
public:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> implementation);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  [[nodiscard]] static std::shared_ptr<Model> Load(const std::string& path,
                                                   std::uint32_t context);
  [[nodiscard]] std::unique_ptr<Session> CreateSession() const;
  [[nodiscard]] const Config& config() const;
  [[nodiscard]] const Tokenizer& tokenizer() const;
  [[nodiscard]] std::string Name() const;
  [[nodiscard]] std::size_t WeightBytes() const;
  [[nodiscard]] std::size_t ScratchBytes() const;
  [[nodiscard]] std::uint32_t Context() const;
  void LoadAssistant(const std::string& path);
  [[nodiscard]] bool HasAssistant() const;
  void LoadVision(const std::string& path);
  [[nodiscard]] bool HasVision() const;
  [[nodiscard]] std::size_t KvBytes() const;
  [[nodiscard]] const std::string& VisionIdentity() const;
  [[nodiscard]] EncodedImage EncodeImage(std::span<const std::uint8_t> bytes,
                                         const std::string& directory = {});
  [[nodiscard]] EncodedImage EncodePixels(
      const core::Image& pixels, const std::function<bool()>& cancelled);

private:
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

class Session {
public:
  struct Impl;
  explicit Session(std::unique_ptr<Impl> implementation);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  void Evaluate(std::uint32_t token);
  static constexpr std::size_t kPrefillBatch = 128;
  void EvaluateTokens(std::span<const std::uint32_t> tokens);
  void EvaluateImage(std::span<const float> embeddings);
  void Reset() noexcept;
  [[nodiscard]] std::size_t Position() const;
  [[nodiscard]] std::span<const float> Logits() const;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const;
  [[nodiscard]] std::unique_ptr<Snapshot> Save() const;
  void Restore(const Snapshot& snapshot);
  // Drafts read a frozen target frontier; they never write its KV cache.
  [[nodiscard]] std::vector<std::uint32_t> Draft(std::uint32_t seed,
                                                 std::size_t count);
  void SetAssistantDirectory(std::string directory);
  // Numerical diagnostics remain package-owned. Empty disables layer dumps.
  void SetLayerDirectory(std::string directory);
  void SetLogitObserver(
      std::function<void(std::size_t, std::span<const float>)> observer);
  void SetCancellationCheck(std::function<bool()> check);

private:
  void EvaluateBatch(std::span<const std::uint32_t> tokens,
                     std::span<const float> embeddings);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::gemma4
#endif  // GUFO_MODELS_GEMMA4_MODEL_HPP_
