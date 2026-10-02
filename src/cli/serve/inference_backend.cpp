#include "src/cli/serve/inference_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/common/chat.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/common/register_packages.hpp"
#include "src/models/common/registry.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/deepseek_v4_flash/serve_runner.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/serve_runner.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/serve_runner.hpp"
#endif

namespace gufo::server {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)

bool DiskCacheEnabled(const TextDiskCacheConfig& config) noexcept {
  return !config.directory.empty();
}

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

/// Derives the content identity that keys persistent continuations for one
/// GGUF artifact. Full digests are cached by the open file metadata.
bool FingerprintArtifact(std::string_view label, const core::GgufReader& reader,
                         std::string* fingerprint, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  Logger::Info("loader", "event=load_phase phase=artifact_identity model=" +
                             std::string(label));
  try {
    *fingerprint = core::GgufIdentityHex(reader);
  } catch (const std::exception& exception) {
    SetError(error, std::string("Failed to fingerprint ") + std::string(label) +
                        " GGUF: " + exception.what());
    return false;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  std::ostringstream message;
  message << "Fingerprinted " << label << " artifact ("
          << core::kGgufIdentityScheme << ", " << reader.GetTensorCount()
          << " tensors) in " << std::fixed << std::setprecision(0) << elapsed_ms
          << " ms";
  Logger::Info("engine", message.str());
  return true;
}

bool FingerprintArtifactFile(std::string_view label,
                             const std::filesystem::path& path,
                             std::string* fingerprint, std::string* error) {
  std::string open_error;
  const auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (reader == nullptr) {
    SetError(error, std::string("Failed to open ") + std::string(label) +
                        " GGUF for fingerprinting: " + open_error);
    return false;
  }
  return FingerprintArtifact(label, *reader, fingerprint, error);
}

models::common::SpeculativeRequest ToCommonSpeculative(
    const TextSpeculativeConfig& config) {
  models::common::SpeculativeRequest request;
  switch (config.backend) {
    case TextSpeculativeBackend::kDisabled:
      request.backend = models::common::SpeculativeBackend::kDisabled;
      break;
    case TextSpeculativeBackend::kDFlash:
      request.backend = models::common::SpeculativeBackend::kDFlash;
      break;
    case TextSpeculativeBackend::kDSpark:
      request.backend = models::common::SpeculativeBackend::kDSpark;
      break;
    case TextSpeculativeBackend::kMtp:
      request.backend = models::common::SpeculativeBackend::kMtp;
      break;
  }
  request.draft_model_path = config.draft_model_path;
  request.max_draft_tokens = config.max_draft_tokens;
  request.min_draft_tokens = config.min_draft_tokens;
  request.draft_policy =
      config.dflash_policy == speculative::DFlashDraftPolicy::kFixed
          ? models::common::DraftLengthPolicy::kFixed
          : models::common::DraftLengthPolicy::kAdaptive;
  request.mtp_survival = config.mtp_survival;
  request.mtp_latin_draft_vocabulary = config.mtp_latin_draft_vocabulary;
  request.prompt_lookup = config.prompt_lookup;
  return request;
}

constexpr std::string_view DraftFingerprintLabel(
    TextSpeculativeBackend backend) noexcept {
  switch (backend) {
    case TextSpeculativeBackend::kDFlash:
      return "DFlash";
    case TextSpeculativeBackend::kDSpark:
      return "DSpark";
    case TextSpeculativeBackend::kMtp:
      return "MTP";
    case TextSpeculativeBackend::kDisabled:
      return "draft";
  }
  return "draft";
}

#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_image_input{false};
    ReasoningOptions reasoning_defaults;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent)
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)) {
      if (initial == InitialOutputState::kReasoning) {
        const std::string think_end =
            state_->scheduler->runner().Descriptor().output_dialect.think_end;
        if (!think_end.empty())
          reasoning_end_ = state_->scheduler->runner().Tokenize(think_end);
      }
    }

    Result Wait(const TokenCallback& on_token) override {
      auto result = request_.Wait(on_token);
      if (!reasoning_end_.empty()) {
        const auto end =
            std::search(result.tokens.begin(), result.tokens.end(),
                        reasoning_end_.begin(), reasoning_end_.end());
        result.reasoning_tokens =
            static_cast<std::size_t>(end - result.tokens.begin());
      }
      return result;
    }

    void Cancel() noexcept override { request_.Cancel(); }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    std::vector<tokenization::TokenId> reasoning_end_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token);
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path) {
#if defined(ENGINE_ENABLE_HIP)
  TextDiskCacheConfig resolved_disk_cache_config = disk_cache_config;
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  models::common::RegisterAllModelPackages();
  const models::common::TextModelPackage* package =
      models::common::TextModelRegistry::Global().FindForReader(*reader);
  if (package == nullptr) {
    SetError(error,
             "No compiled-in model package handles architecture '" +
                 std::string(reader->GetMetadataString("general.architecture")
                                 .value_or("")) +
                 "'");
    return false;
  }
  if (max_context == 0) {
    max_context = package->NativeContext(*reader);
    if (max_context == 0) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return false;
    }
  }
  if (!package->ValidateTemplate(*reader, &load_error)) {
    SetError(error, load_error);
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
      !FingerprintArtifact(
          package->Name(), *reader,
          &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      !speculative_config.draft_model_path.empty() &&
      resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
      !FingerprintArtifactFile(
          DraftFingerprintLabel(speculative_config.backend),
          speculative_config.draft_model_path,
          &resolved_disk_cache_config.draft_model_artifact_fingerprint,
          error)) {
    return false;
  }
  models::common::TextModelLoadOptions load_options;
  load_options.model_path = model_path;
  load_options.max_context = max_context;
  load_options.session_count = session_count;
  load_options.speculative = ToCommonSpeculative(speculative_config);
  load_options.vision_model_path = vision_model_path;
  load_options.model_artifact_fingerprint =
      resolved_disk_cache_config.model_artifact_fingerprint;
  load_options.draft_model_artifact_fingerprint =
      resolved_disk_cache_config.draft_model_artifact_fingerprint;
  load_options.disk_cache_enabled =
      DiskCacheEnabled(resolved_disk_cache_config);
  load_options.disk_cache_capacity_bytes =
      resolved_disk_cache_config.capacity_bytes;
  auto loaded = package->Load(load_options, &load_error);
  if (loaded == nullptr || loaded->runner == nullptr) {
    SetError(error, load_error);
    return false;
  }
  return InstallRunner(std::move(loaded->runner), loaded->model_id,
                       loaded->supports_image_input, loaded->max_context,
                       session_count, prefill_policy, scheduler_policy,
                       std::move(resolved_disk_cache_config), error);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  (void)vision_model_path;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::InstallRunner(
    std::shared_ptr<TextModelRunner> runner, const std::string& model_id,
    bool supports_image_input, std::uint32_t max_context,
    std::size_t session_count, TextPrefillPolicy prefill_policy,
    TextSchedulerPolicy scheduler_policy, TextDiskCacheConfig disk_cache_config,
    std::string* error) {
  if (runner == nullptr) {
    SetError(error, "HTTP runner must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->supports_image_input = supports_image_input;
    new_state->model_id = model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache));
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config,
                            TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->GetConfig().context_length;
  if (max_context < 2 || max_context > model->GetConfig().context_length) {
    SetError(error, "HTTP context exceeds the loaded Qwen model context");
    return false;
  }

  if (speculative_config.backend == TextSpeculativeBackend::kDSpark) {
    SetError(error, "DSpark HTTP decoding requires a DeepSeek model");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
    SetError(error, "MTP HTTP decoding requires a Qwen Flash-Next model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (!disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "Qwen persistent disk cache configuration is invalid");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash &&
      speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DFlash2 requires --min-draft-tokens 1; bound blocks with "
             "--draft-tokens");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    speculative::SpeculativeOptions speculative_options;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          disk_cache_config.draft_model_artifact_fingerprint.empty() &&
          !FingerprintArtifactFile(
              "DFlash", speculative_config.draft_model_path,
              &disk_cache_config.draft_model_artifact_fingerprint, error)) {
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) {
        SetError(error,
                 "Qwen DFlash persistent disk cache configuration is "
                 "invalid");
        return false;
      }
      dflash_model = models::qwen::serve::LoadDflashModel(
          model, speculative_config.draft_model_path,
          speculative_config.max_draft_tokens,
          speculative_config.min_draft_tokens, speculative_config.dflash_policy,
          &speculative_options, error);
      if (dflash_model == nullptr) {
        return false;
      }
    }

    const bool supports_image = model->VisionEncoder() != nullptr;
    auto runner = models::qwen::serve::CreateTextRunner(
        std::move(model), max_context, std::move(dflash_model),
        speculative_options, disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    const std::string model_id = runner->Descriptor().model_id;
    return InstallRunner(std::move(runner), model_id, supports_image,
                         max_context, session_count, prefill_policy,
                         scheduler_policy, std::move(disk_cache_config), error);
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  const bool use_dspark =
      speculative_config.backend == TextSpeculativeBackend::kDSpark;
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (!use_dspark || !model->HasDspark())) {
    SetError(error, "DeepSeek DSpark requires a loaded support model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }
  if (use_dspark && (speculative_config.max_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens >
                         speculative_config.max_draft_tokens)) {
    SetError(error, "DeepSeek DSpark draft limits are invalid");
    return false;
  }
  if (use_dspark && speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DSpark uses model-owned adaptive drafting; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (use_dspark &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "DeepSeek persistent disk cache configuration is invalid");
    return false;
  }

  try {
    auto runner = models::deepseek_v4_flash::serve::CreateTextRunner(
        std::move(model), max_context, use_dspark,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    const std::string model_id = runner->Descriptor().model_id;
    return InstallRunner(std::move(runner), model_id,
                         /*supports_image_input=*/false, max_context,
                         session_count, prefill_policy, scheduler_policy,
                         std::move(disk_cache_config), error);
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen3.8-Flash-Next model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(
        error,
        "HTTP context exceeds the loaded Qwen3.8-Flash-Next model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Flash-Next MTP requires a model loaded with its draft sidecar");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      (speculative_config.max_draft_tokens == 0 ||
       speculative_config.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.8-Flash-Next MTP drafts a fixed chain; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error,
             "Qwen3.8-Flash-Next persistent disk cache configuration is "
             "invalid");
    return false;
  }
  try {
    const bool supports_image = model->VisionEncoder() != nullptr;
    auto runner = models::qwen38_flash_next::serve::CreateTextRunner(
        std::move(model), max_context,
        speculative_config.backend == TextSpeculativeBackend::kMtp,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    const std::string model_id = runner->Descriptor().model_id;
    return InstallRunner(std::move(runner), model_id, supports_image,
                         max_context, session_count, prefill_policy,
                         scheduler_policy, std::move(disk_cache_config), error);
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

bool InferenceBackend::supports_image_input() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_image_input;
#else
  return false;
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

models::common::OutputDialect InferenceBackend::output_dialect() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().Descriptor().output_dialect
             : models::common::OutputDialect{};
#else
  return {};
#endif
}

void InferenceBackend::set_model_id(const std::string& model_id) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_id.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = model_id;
  impl_->state = std::move(updated);
#else
  (void)model_id;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults = {
      .max_tokens = max_tokens,
      .sampling = sampling_config,
  };
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id,
    const std::vector<std::string>& stop_sequences) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {}, true,
      0, stop_sequences);
#else
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt = state->scheduler->runner().PreparePrompt(request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences,
      state->scheduler->runner().InitialOutputState(request));
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto prompt = state->scheduler->runner().PreparePrompt(request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request),
      state->scheduler->runner().InitialOutputState(request));
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<models::common::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

}  // namespace gufo::server
