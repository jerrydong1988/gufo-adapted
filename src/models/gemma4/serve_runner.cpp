#include "src/models/gemma4/serve_runner.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/image.hpp"
#include "src/models/common/model_package.hpp"
#include "src/models/common/registry.hpp"
#include "src/models/gemma4/model.hpp"
#include "src/models/gemma4/vision.hpp"

namespace gufo::models::gemma4 {
namespace {

struct ImageContext final : server::TextPromptContext {
  struct Input {
    core::Image pixels;
    std::size_t begin, end;
    std::vector<std::uint8_t> identity;
  };
  std::vector<Input> images;
  std::span<const std::uint8_t> IdentityBefore(
      std::size_t count) const override {
    for (auto i = images.rbegin(); i != images.rend(); ++i)
      if (i->begin < count)
        return i->identity;
    return {};
  }
  std::size_t PrefillBoundary(std::size_t requested) const override {
    for (const auto& image : images)
      if (image.begin < requested && requested < image.end)
        return image.end;
    return requested;
  }
};

server::TextPreparedPrompt Prepare(const Model& model,
                                   const server::ChatRequest& request) {
  if (!request.tools.empty() ||
      request.tool_choice == server::ChatRequest::ToolChoice::kRequired)
    throw std::invalid_argument("Gemma 4 tool requests are not supported");
  std::vector<RenderedImage> attachments;
  const auto rendered =
      RenderChat(request.messages, request.reasoning, &attachments);
  if (attachments.empty())
    return {model.tokenizer().Encode(rendered, false), {}};
  if (!model.HasVision())
    throw std::invalid_argument(
        "Gemma image input requires its compatible --mmproj BF16 projector");
  auto context = std::make_shared<ImageContext>();
  server::TextPreparedPrompt result;
  crypto::Sha256Hasher hasher;
  const std::string scheme =
      "gemma4-image-rgb8-pillow-48-v1:" + model.VisionIdentity();
  hasher.Update(
      {reinterpret_cast<const std::uint8_t*>(scheme.data()), scheme.size()});
  const auto number = [&](std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = value >> (i * 8);
    hasher.Update(bytes);
  };
  constexpr std::string_view marker = "<|image|>";
  std::size_t cursor = 0;
  for (const auto& attachment : attachments) {
    const auto segment = model.tokenizer().Encode(
        std::string_view(rendered).substr(cursor, attachment.offset - cursor),
        false);
    result.tokens.insert(result.tokens.end(), segment.begin(), segment.end());
    result.tokens.push_back(255999);  // <|image>
    const auto begin = result.tokens.size();
    auto pixels = PreprocessImage(core::DecodeImage(*attachment.bytes));
    const auto count = std::size_t(pixels.width / 48) * (pixels.height / 48);
    result.tokens.insert(result.tokens.end(), count, 258880);
    const auto end = result.tokens.size();
    result.tokens.push_back(258882);  // <image|>
    number(begin);
    number(end);
    number(pixels.width);
    number(pixels.height);
    number(pixels.pixels.size());
    hasher.Update(pixels.pixels);
    const auto digest = hasher.Digest();
    context->images.push_back(
        {std::move(pixels), begin, end, {digest.begin(), digest.end()}});
    cursor = attachment.offset + marker.size();
  }
  const auto suffix = model.tokenizer().Encode(
      std::string_view(rendered).substr(cursor), false);
  result.tokens.insert(result.tokens.end(), suffix.begin(), suffix.end());
  if (result.tokens.size() >= model.Context())
    throw std::invalid_argument(
        "Gemma image prompt exceeds the bounded context");
  context->cache_identity = context->images.back().identity;
  result.context = std::move(context);
  return result;
}

class State final : public server::TextRunnerState {
public:
  explicit State(std::unique_ptr<Session> value) : session(std::move(value)) {}
  void Invalidate() noexcept override {
    session->Reset();
    context.reset();
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session->SetCancellationCheck(check);
    cancelled = check;
  }
  std::unique_ptr<Session> session;
  std::shared_ptr<const ImageContext> context;
  CancellationCheck cancelled;
};

class RunnerSnapshot final : public server::TextRunnerSnapshot {
public:
  explicit RunnerSnapshot(std::unique_ptr<gemma4::Snapshot> value)
      : value(std::move(value)) {}
  std::size_t PayloadBytes() const noexcept override {
    return value->PayloadBytes();
  }
  std::unique_ptr<gemma4::Snapshot> value;
};

class Runner final : public server::TextModelRunner {
public:
  explicit Runner(std::shared_ptr<Model> model, std::size_t draft_tokens)
      : model_(std::move(model)), draft_tokens_(draft_tokens) {}
  server::TextRunnerDescriptor Descriptor() const override {
    server::TextRunnerDescriptor result;
    result.model_id = model_->Name();
    result.state_abi = model_->HasVision()
                           ? "gemma4-31b-fp32-image-batch-ring-v2"
                           : "gemma4-31b-fp32-batch-ring-v2";
    result.max_context = model_->Context();
    result.capabilities.incremental_prefill = true;
    result.capabilities.incremental_text_is_exact = true;
    result.capabilities.prefix_reuse = true;
    result.capabilities.snapshot = true;
    result.capabilities.fork = true;
    result.capabilities.multi_token_decode = model_->HasAssistant();
    result.output_dialect = {.think_start = "<|channel>thought\n",
                             .think_end = "<channel|>",
                             .qwen_tool_calls = false,
                             .dsml_tool_calls = false};
    return result;
  }
  server::TextRunnerResourceClaim ResourceClaim() const override {
    const auto bytes = model_->KvBytes() + (model_->config().hidden +
                                            2 * model_->config().vocabulary) *
                                               sizeof(float);
    return {
        .resident_weights_bytes = model_->WeightBytes(),
        .state_capacity_bytes = bytes + model_->ScratchBytes(),
        .per_request_state_bytes = bytes,
        .temporary_scratch_bytes = model_->ScratchBytes(),
        .retained_snapshot_capacity_bytes = server::HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true};
  }
  std::vector<server::TextExecutionPlan> SupportedPlans() const override {
    return {{}};
  }
  std::vector<server::TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }
  std::optional<std::vector<server::TextRunnerToken>> RenderAndTokenize(
      const server::ChatRequest& request) const override {
    for (const auto& message : request.messages)
      if (!message.images.empty())
        throw std::invalid_argument(
            "Gemma images require PreparePrompt and its image context");
    if (!request.tools.empty() ||
        request.tool_choice == server::ChatRequest::ToolChoice::kRequired)
      throw std::invalid_argument("Gemma 4 tool requests are not supported");
    return model_->tokenizer().Encode(
        RenderChat(request.messages, request.reasoning), false);
  }
  std::optional<server::TextPreparedPrompt> PreparePrompt(
      const server::ChatRequest& request) const override {
    return Prepare(*model_, request);
  }
  void SetPromptContext(
      server::TextRunnerState& opaque,
      std::shared_ptr<const server::TextPromptContext> context) const override {
    auto& state = dynamic_cast<State&>(opaque);
    const auto image = std::dynamic_pointer_cast<const ImageContext>(context);
    if (context && !image)
      throw std::invalid_argument("incompatible Gemma image prompt context");
    state.context = image;
  }
  server::TextGenerationBackend::InitialOutputState InitialOutputState(
      const server::ChatRequest&) const override {
    return server::TextGenerationBackend::InitialOutputState::kAuto;
  }
  std::string Decode(
      std::span<const server::TextRunnerToken> ids) const override {
    return model_->tokenizer().Decode(ids);
  }
  std::unique_ptr<server::TextRunnerState> CreateState() const override {
    return std::make_unique<State>(model_->CreateSession());
  }
  server::TextPrefillStep Prefill(
      server::TextRunnerState& opaque,
      std::span<const server::TextRunnerToken> prompt, std::size_t offset,
      std::size_t maximum) const override {
    auto& state = dynamic_cast<State&>(opaque);
    if (offset != state.session->Position() || offset >= prompt.size() ||
        maximum == 0)
      throw std::invalid_argument("invalid Gemma 4 prefill frontier");
    if (state.context)
      for (const auto& image : state.context->images) {
        if (offset == image.begin) {
          const auto count = image.end - image.begin;
          if (maximum < count || image.end > prompt.size() ||
              !std::ranges::all_of(prompt.subspan(offset, count),
                                   [](auto token) { return token == 258880; }))
            throw std::invalid_argument(
                "Gemma bidirectional image block must be prefetched "
                "atomically");
          const auto encoded =
              model_->EncodePixels(image.pixels, state.cancelled);
          if (encoded.Tokens() != count)
            throw std::runtime_error("Gemma encoder token count changed");
          state.session->EvaluateImage(encoded.embeddings);
          return {count, image.end == prompt.size()};
        }
        if (image.begin < offset && offset < image.end)
          throw std::invalid_argument(
              "Gemma cannot resume inside an image block");
      }
    auto count =
        std::min({maximum, prompt.size() - offset, Session::kPrefillBatch});
    if (state.context)
      for (const auto& image : state.context->images)
        if (image.begin > offset)
          count = std::min(count, image.begin - offset);
    state.session->EvaluateTokens(prompt.subspan(offset, count));
    return {count, offset + count == prompt.size()};
  }
  server::TextDecodeSelection SelectNext(
      server::TextRunnerState& opaque,
      sampling::SamplerState& sampler) const override {
    auto& state = dynamic_cast<State&>(opaque);
    const auto token = sampler.Sample(state.session->Logits());
    return {Tokenizer::Stop(token), token,
            model_->tokenizer().Decode(std::span(&token, 1))};
  }
  void Advance(server::TextRunnerState& opaque,
               server::TextRunnerToken token) const override {
    dynamic_cast<State&>(opaque).session->Evaluate(token);
  }
  server::TextDecodeStep DecodeStep(
      server::TextRunnerState& opaque, std::size_t maximum,
      sampling::SamplerState& sampler) const override {
    if (!model_->HasAssistant() || maximum <= 1)
      return TextModelRunner::DecodeStep(opaque, maximum, sampler);
    auto& session = *dynamic_cast<State&>(opaque).session;
    auto tentative = sampler;
    server::TextDecodeStep result;
    auto seed = tentative.Sample(session.Logits());
    if (Tokenizer::Stop(seed)) {
      sampler.CopyDrawStateFrom(tentative);
      result.stop = true;
      return result;
    }
    const auto remaining = model_->Context() - session.Position();
    const auto count = std::min(
        {draft_tokens_, maximum - 1, remaining > 0 ? remaining - 1 : 0});
    // A deterministic draft is a point-mass proposal. Sampling the target and
    // accepting iff it equals that proposal preserves the exact target law,
    // including filters/penalties and the non-speculative RNG sequence.
    const auto proposals = session.Draft(seed, count);
    result.draft_tokens = proposals.size();
    try {
      session.Evaluate(seed);
      result.selections.push_back(
          {false, seed, model_->tokenizer().Decode(std::span(&seed, 1))});
      tentative.Accept(seed);
      sampler.CopyDrawStateFrom(tentative);
      for (const auto proposal : proposals) {
        const auto token = tentative.Sample(session.Logits());
        if (Tokenizer::Stop(token)) {
          sampler.CopyDrawStateFrom(tentative);
          result.stop = true;
          break;
        }
        session.Evaluate(token);
        result.selections.push_back(
            {false, token, model_->tokenizer().Decode(std::span(&token, 1))});
        tentative.Accept(token);
        sampler.CopyDrawStateFrom(tentative);
        if (token != proposal)
          break;
        ++result.draft_accepted_tokens;
      }
    } catch (...) {
      result.failure = std::current_exception();
    }
    return result;
  }
  std::size_t CheckpointPosition(
      const server::TextRunnerState& opaque) const override {
    return dynamic_cast<const State&>(opaque).session->Position();
  }
  std::size_t SnapshotPayloadBytes(
      const server::TextRunnerState& opaque) const override {
    return dynamic_cast<const State&>(opaque).session->SnapshotPayloadBytes();
  }
  std::unique_ptr<server::TextRunnerSnapshot> Snapshot(
      const server::TextRunnerState& opaque) const override {
    return std::make_unique<RunnerSnapshot>(
        dynamic_cast<const State&>(opaque).session->Save());
  }
  void RestoreOrFork(
      server::TextRunnerState& opaque,
      const server::TextRunnerSnapshot& snapshot) const override {
    const auto* source = dynamic_cast<const RunnerSnapshot*>(&snapshot);
    if (!source)
      throw std::invalid_argument("incompatible Gemma 4 snapshot");
    dynamic_cast<State&>(opaque).session->Restore(*source->value);
  }

private:
  std::shared_ptr<Model> model_;
  std::size_t draft_tokens_;
};

class Package final : public common::TextModelPackage {
public:
  std::string Name() const override { return "gemma4"; }
  std::vector<std::string> Architectures() const override { return {"gemma4"}; }
  bool ValidateTemplate(const core::GgufReader& reader,
                        std::string* error) const override {
    return gemma4::ValidateTemplate(reader, error);
  }
  std::uint32_t NativeContext(const core::GgufReader&) const override {
    return 4096;
  }
  bool ValidateLoadOptions(const common::TextModelLoadOptions& options,
                           std::string* error) const override {
    const auto reject = [&](const char* message) {
      if (error)
        *error = message;
      return false;
    };
    if (options.max_context < 2 || options.max_context > 4096)
      return reject("Gemma 4 currently supports 2..4096 context tokens");
    if (options.session_count != 1)
      return reject("Gemma 4 currently supports one session");
    if (options.speculative.prompt_lookup ||
        options.speculative.backend == common::SpeculativeBackend::kDFlash ||
        options.speculative.backend == common::SpeculativeBackend::kDSpark)
      return reject("Gemma 4 supports only its assistant MTP backend");
    if (options.speculative.backend == common::SpeculativeBackend::kMtp) {
      if (options.speculative.draft_model_path.empty() ||
          options.speculative.max_draft_tokens < 1 ||
          options.speculative.max_draft_tokens > 7 ||
          options.speculative.min_draft_tokens != 1)
        return reject(
            "Gemma MTP requires its matching assistant, 1..7 drafts and "
            "minimum draft count 1");
      if (options.speculative.mtp_survival ||
          options.speculative.mtp_latin_draft_vocabulary)
        return reject(
            "Gemma MTP does not support Flash-Next controller options");
    } else if (!options.speculative.draft_model_path.empty())
      return reject("Gemma assistant requires --speculative mtp");
    if (options.disk_cache_enabled)
      return reject("Gemma 4 persistent snapshots are not yet supported");
    return true;
  }
  std::unique_ptr<common::LoadedTextModel> Load(
      const common::TextModelLoadOptions& options,
      std::string* error) const override {
    if (!ValidateLoadOptions(options, error))
      return nullptr;
    try {
      auto model = Model::Load(options.model_path, options.max_context);
      if (options.speculative.backend == common::SpeculativeBackend::kMtp)
        model->LoadAssistant(options.speculative.draft_model_path);
      if (!options.vision_model_path.empty())
        model->LoadVision(options.vision_model_path);
      auto result = std::make_unique<common::LoadedTextModel>();
      result->model_id = model->Name();
      result->max_context = model->Context();
      result->supports_image_input = model->HasVision();
      result->runner = CreateTextRunner(std::move(model),
                                        options.speculative.max_draft_tokens);
      return result;
    } catch (const std::exception& failure) {
      if (error)
        *error = failure.what();
      return nullptr;
    }
  }
};

}  // namespace

std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::size_t draft_tokens) {
  return std::make_shared<Runner>(std::move(model), draft_tokens);
}
std::span<const float> SnapshotLogits(
    const server::TextRunnerSnapshot& snapshot) {
  return dynamic_cast<const RunnerSnapshot&>(snapshot).value->Logits();
}
std::string SnapshotDigest(const server::TextRunnerSnapshot& snapshot) {
  return dynamic_cast<const RunnerSnapshot&>(snapshot).value->Digest();
}
void RegisterGemma4Package() {
  common::TextModelRegistry::Global().Register(std::make_shared<Package>());
}

}  // namespace gufo::models::gemma4
