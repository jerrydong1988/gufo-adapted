#include "src/models/qwen38_flash_next/serve_runner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/device_memory.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/common/registry.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/serve_prompt.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

// Qwen3.8-Flash-Next TextModelRunner adapter and model package. The runner and
// snapshot code below moved verbatim from the inference backend; only the
// shared prompt-helper qualifications changed.

namespace gufo::models::qwen38_flash_next::serve {

namespace detail {

using namespace gufo::server;

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
  }
  if (has_mtp && !IsSha256Hex(mtp_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v4\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v"
           << models::qwen38_flash_next::Session::kSnapshotPayloadVersion
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

// Each Flash-Next request owns its recurrent/KV state and snapshots. Target
// projections share batches; sampling and rollback remain request-local.
constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(const std::shared_ptr<QwenFlashNextModel>& model,
                               std::uint32_t max_context, bool use_mtp) {
    std::string error;
    session_ =
        model->CreateSession(use_mtp ? gufo::core::SessionMode::kSpeculative
                                     : gufo::core::SessionMode::kAutoregressive,
                             max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Qwen3.8-Flash-Next session: " +
                               error);
    }
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Reset();
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<QwenFlashNextSession> session_;
  std::size_t position_{0};
};

class QwenFlashNextTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenFlashNextTextRunnerSnapshot(
      std::shared_ptr<QwenFlashNextModel> model,
      std::unique_ptr<models::qwen38_flash_next::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<QwenFlashNextModel> model;
  std::unique_ptr<models::qwen38_flash_next::SessionSnapshot> snapshot;
  std::size_t position;
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

class QwenFlashNextTextRunner final : public TextModelRunner {
public:
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    if (!artifact_fingerprint.empty()) {
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenFlashNextCompatibilityIdentity(
              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
              use_mtp_, max_context_, max_draft_tokens_,
              model_->DecodeConcurrency()),
          .payload_version =
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwenFlashNextStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
        // Flash-Next emits <think> reasoning with Qwen tool-call markup;
        // the shared parser keeps DSML detection enabled for compatibility.
        .output_dialect = models::common::OutputDialect{},
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (gufo::platform::DeviceMemoryInfo(&free_bytes, &total_bytes) ==
        hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Snapshots live in host memory, not in the device state pool.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? gufo::core::SessionMode::kSpeculative
                     : gufo::core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        models::qwen::serve::QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return models::qwen::serve::PrepareQwenPrompt(
        request, model_->tokenizer(), model_->VisionEncoder(), max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenFlashNextState(state).session().ConfigureVision(
        models::qwen::serve::QwenPrompt(context));
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return models::qwen::serve::QwenChatOptions(request, max_context_)
                   .enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenFlashNextTextRunnerState>(model_, max_context_,
                                                          use_mtp_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    std::string error;
    if (!qfn.session().Sync(prefix, &error)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    if (!qfn.session().DecodeStep(budget, working_sampler, &decoded, &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    step.lookup_tokens = stats_after.lookup - stats_before.lookup;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& session =
          RequireQwenFlashNextState(decodes[i].state.get()).session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], true, &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      step.stop = results[i].stop;
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      for (const auto token : results[i].tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      const auto stats = state.session().Statistics();
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const std::uint64_t bytes =
        RequireQwenFlashNextState(state).session().SnapshotBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(
          "Qwen3.8-Flash-Next snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    auto snapshot = qfn.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("Qwen3.8-Flash-Next snapshot failed: " + error);
    }
    return std::make_unique<QwenFlashNextTextRunnerSnapshot>(
        model_, std::move(snapshot), qfn.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next snapshot does not belong to this model");
    }
    auto& restored = RequireQwenFlashNextState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*qfn_snapshot->snapshot, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Qwen3.8-Flash-Next snapshot restore failed: " +
                               error);
    }
    restored.set_position(qfn_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next persistent snapshot does not belong to this "
          "model");
    }
    return qfn_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr ||
        destination.size() != qfn_snapshot->PayloadBytes() ||
        !qfn_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const QwenFlashNextTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireQwenFlashNextState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error(
          "Qwen3.8-Flash-Next persistent snapshot restore failed: " + error);
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(
          "Qwen3.8-Flash-Next persistent snapshot restored an invalid "
          "position");
    }
    restored.set_position(position);
  }

private:
  std::shared_ptr<QwenFlashNextModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

}  // namespace detail

namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::shared_ptr<::gufo::server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::string artifact_fingerprint,
    std::string mtp_fingerprint) {
  return std::make_shared<detail::QwenFlashNextTextRunner>(
      std::move(model), max_context, use_mtp, max_draft_tokens,
      std::move(artifact_fingerprint), std::move(mtp_fingerprint));
}

class FlashNextTextModelPackage final : public common::TextModelPackage {
public:
  [[nodiscard]] std::string Name() const override { return "qwen4exp"; }
  [[nodiscard]] std::vector<std::string> Architectures() const override {
    return {"qwen4exp"};
  }
  [[nodiscard]] bool ValidateTemplate(const core::GgufReader& reader,
                                      std::string* error) const override {
    std::string template_error;
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(
            reader, &template_error)) {
      SetError(error, "Unsupported Qwen3.8-Flash-Next chat template: " +
                          template_error);
      return false;
    }
    return true;
  }
  [[nodiscard]] std::uint32_t NativeContext(
      const core::GgufReader& reader) const override {
    const std::string architecture = std::string(
        reader.GetMetadataString("general.architecture").value_or(""));
    const std::uint64_t native =
        reader.GetMetadataUint64(architecture + ".context_length").value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      return 0;
    }
    return static_cast<std::uint32_t>(native);
  }
  [[nodiscard]] bool ValidateLoadOptions(
      const common::TextModelLoadOptions& options,
      std::string* error) const override {
    const common::SpeculativeRequest& spec = options.speculative;
    const bool use_mtp = spec.backend == common::SpeculativeBackend::kMtp;
    if (spec.backend != common::SpeculativeBackend::kDisabled && !use_mtp) {
      SetError(error,
               "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
               "decoding (--speculative mtp --mtp-model)");
      return false;
    }
    if (use_mtp && spec.draft_model_path.empty()) {
      SetError(error,
               "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (use_mtp && (spec.max_draft_tokens == 0 || spec.min_draft_tokens != 1)) {
      SetError(error,
               "Flash-Next MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    if (options.disk_cache_enabled &&
        (!detail::IsSha256Hex(options.model_artifact_fingerprint) ||
         (use_mtp &&
          !detail::IsSha256Hex(options.draft_model_artifact_fingerprint)) ||
         options.disk_cache_capacity_bytes == 0)) {
      SetError(error,
               "Qwen3.8-Flash-Next persistent disk cache configuration is "
               "invalid");
      return false;
    }
    return true;
  }
  [[nodiscard]] std::unique_ptr<common::LoadedTextModel> Load(
      const common::TextModelLoadOptions& options,
      std::string* error) const override {
    if (!ValidateLoadOptions(options, error)) {
      return nullptr;
    }
    if (options.max_context == 0) {
      SetError(error, "Flash-Next load requires a resolved context length");
      return nullptr;
    }
    try {
      const bool use_mtp =
          options.speculative.backend == common::SpeculativeBackend::kMtp;
      std::string load_error;
      // The model owns prefill geometry for both bulk and scheduled requests.
      auto model = Model::Load(
          options.model_path,
          ModelOptions{
              .max_context = options.max_context,
              .mtp_model_path = use_mtp ? options.speculative.draft_model_path
                                        : std::string{},
              .max_draft_tokens = options.speculative.max_draft_tokens,
              .vision_model_path = options.vision_model_path,
              .decode_concurrency = static_cast<std::uint32_t>(
                  std::clamp<std::size_t>(options.session_count, 1, 8)),
              .mtp_survival = options.speculative.mtp_survival,
              .prompt_lookup = options.speculative.prompt_lookup,
              .draft_vocabulary = options.speculative.mtp_latin_draft_vocabulary
                                      ? DraftVocabulary::kLatinText
                                      : DraftVocabulary::kFull,
          },
          &load_error);
      if (model == nullptr) {
        SetError(error,
                 "Failed to create Qwen3.8-Flash-Next model: " + load_error);
        return nullptr;
      }
      auto runner = CreateTextRunner(model, options.max_context, use_mtp,
                                     options.speculative.max_draft_tokens,
                                     options.model_artifact_fingerprint,
                                     options.draft_model_artifact_fingerprint);
      auto loaded = std::make_unique<common::LoadedTextModel>();
      loaded->model_id = runner->Descriptor().model_id;
      loaded->supports_image_input = model->VisionEncoder() != nullptr;
      loaded->max_context = options.max_context;
      loaded->runner = std::move(runner);
      return loaded;
    } catch (const std::exception& exception) {
      SetError(error, exception.what());
      return nullptr;
    }
  }
};

void RegisterFlashNextPackage() {
  common::TextModelRegistry::Global().Register(
      std::make_shared<FlashNextTextModelPackage>());
}

}  // namespace gufo::models::qwen38_flash_next::serve
