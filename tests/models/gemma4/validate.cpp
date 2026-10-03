#include "src/models/common/validate/validate.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/gemma4/model.hpp"
#include "src/models/gemma4/serve_runner.hpp"
#include "src/models/gemma4/vision.hpp"

int main(int argc, char** argv) {
  try {
    if (argc < 3)
      throw std::invalid_argument(
          "usage: gemma4_validate MODEL TOKENS LOGITS [LAYERS], or MODEL "
          "--harness");
    auto model = gufo::models::gemma4::Model::Load(argv[1], 4096);
    std::cout << "weights=" << model->WeightBytes()
              << " kv=" << model->KvBytes()
              << " scratch=" << model->ScratchBytes() << '\n';
    if (std::string(argv[2]) == "--harness") {
      if (argc > 3)
        model->LoadAssistant(argv[3]);
      const auto report = gufo::models::common::ValidateLoadedModel(
          {.runner = gufo::models::gemma4::CreateTextRunner(model),
           .model_id = model->Name(),
           .max_context = 4096},
          {.prefill_tokens = 16, .decode_tokens = 8});
      for (const auto& check : report.checks)
        std::cout << check.name << ": "
                  << (check.skipped  ? "SKIP"
                      : check.passed ? "PASS"
                                     : "FAIL")
                  << ' ' << check.detail << '\n';
      return report.passed() ? 0 : 1;
    }
    if (std::string(argv[2]) == "--state") {
      if (argc < 4)
        throw std::invalid_argument("missing state token fixture");
      std::ifstream input(argv[3]);
      std::vector<std::uint32_t> fixture;
      for (std::uint32_t token; input >> token;)
        fixture.push_back(token);
      if (fixture.empty())
        throw std::runtime_error("empty state fixture");
      const auto token = [&](std::size_t i) {
        return fixture[i % fixture.size()];
      };
      auto state = model->CreateSession();
      std::map<std::size_t, std::unique_ptr<gufo::models::gemma4::Snapshot>>
          snapshots;
      std::map<std::size_t, std::vector<float>> frontiers;
      for (std::size_t i = 0; i < 2052;) {
        const auto end = i < 20     ? 20
                         : i < 1022 ? 1022
                         : i < 1028 ? 1028
                         : i < 2048 ? 2048
                                    : 2052;
        const auto width = i < 20 || (i >= 1022 && i < 1028) || i >= 2048
                               ? 1
                               : std::min<std::size_t>(128, end - i);
        std::vector<std::uint32_t> batch(width);
        for (std::size_t j = 0; j < width; ++j)
          batch[j] = token(i + j);
        state->EvaluateTokens(batch);
        i += width;
        const auto pos = state->Position();
        if (pos == 20 || (pos >= 1023 && pos <= 1028) || pos >= 2049) {
          frontiers[pos] = std::vector<float>(state->Logits().begin(),
                                              state->Logits().end());
        }
        if (pos == 1023 || pos == 1024 || pos == 1025 || pos == 2049) {
          snapshots[pos] = state->Save();
          if (snapshots[pos]->PayloadBytes() != state->SnapshotPayloadBytes())
            throw std::runtime_error("snapshot byte accounting differs");
          std::cout << "captured=" << pos
                    << " bytes=" << snapshots[pos]->PayloadBytes() << std::endl;
        }
      }
      if (argc > 4) {
        std::ofstream output(argv[4], std::ios::binary);
        for (const auto& [position, row] : frontiers)
          output.write(reinterpret_cast<const char*>(row.data()),
                       row.size() * sizeof(float));
        if (!output)
          throw std::runtime_error("window logits write failed");
      }
      const auto exact = [&](const auto& session) {
        const auto& expected = frontiers.at(session.Position());
        if (!std::equal(expected.begin(), expected.end(),
                        session.Logits().begin()))
          throw std::runtime_error("restored full logits differ at " +
                                   std::to_string(session.Position()));
      };
      for (const auto& [pos, snapshot] : snapshots) {
        state->Restore(*snapshot);
        exact(*state);
        // Restore after two whole windows of writes, then overwrite again.
        for (std::size_t i = pos; i < pos + 3; ++i) {
          state->Evaluate(token(i));
          exact(*state);
        }
        auto fork = model->CreateSession();
        fork->Restore(*snapshot);
        exact(*fork);
        fork->Evaluate(token(pos));
        exact(*fork);
        std::cout << "PASS restore/fork=" << pos << std::endl;
      }
      auto fresh = model->CreateSession();
      for (std::size_t i = 0; i < 20; ++i)
        fresh->Evaluate(token(i));
      exact(*fresh);
      int cancellation_calls = 0;
      state->SetCancellationCheck([&] { return ++cancellation_calls == 20; });
      bool cancelled = false;
      try {
        state->Evaluate(token(state->Position()));
      } catch (const std::runtime_error&) {
        cancelled = true;
      }
      if (!cancelled || state->Position() != 0)
        throw std::runtime_error("cancelled frontier was retained");
      state->SetCancellationCheck({});
      for (std::size_t i = 0; i < 20; ++i)
        state->Evaluate(token(i));
      exact(*state);
      std::cout
          << "PASS cancellation, dirty-tail replacement and fresh recomputation"
          << std::endl;
      return 0;
    }
    if (std::string(argv[2]) == "--image-state") {
      if (argc != 8 && argc != 9)
        throw std::invalid_argument(
            "usage: MODEL --image-state MMPROJ PREFIX EMBEDDINGS SUFFIX LOGITS "
            "[PREFIX_LENGTH]");
      model->LoadVision(argv[3]);
      auto state = model->CreateSession();
      std::ifstream input(argv[4]);
      std::vector<std::uint32_t> prefix;
      for (std::uint32_t token; input >> token;)
        prefix.push_back(token);
      if (prefix.empty())
        throw std::invalid_argument("empty image prefix");
      if (argc == 9) {
        const auto pattern = prefix;
        prefix.resize(std::stoul(argv[8]));
        for (std::size_t i = 0; i < prefix.size(); ++i)
          prefix[i] = pattern[i % pattern.size()];
      }
      for (std::size_t offset = 0; offset < prefix.size();) {
        const auto count = std::min<std::size_t>(128, prefix.size() - offset);
        state->EvaluateTokens(std::span(prefix).subspan(offset, count));
        offset += count;
      }
      std::ifstream image_file(argv[5], std::ios::binary | std::ios::ate);
      const auto bytes = image_file.tellg();
      if (bytes <= 0 || bytes % (5376 * sizeof(float)))
        throw std::invalid_argument("invalid image embedding fixture");
      std::vector<float> embeddings(std::size_t(bytes) / sizeof(float));
      image_file.seekg(0);
      image_file.read(reinterpret_cast<char*>(embeddings.data()), bytes);
      state->EvaluateImage(embeddings);
      auto saved = state->Save();
      std::ofstream output(argv[7], std::ios::binary);
      const auto write = [&] {
        const auto row = state->Logits();
        output.write(reinterpret_cast<const char*>(row.data()),
                     row.size_bytes());
      };
      write();
      std::ifstream suffix(argv[6]);
      std::vector<std::uint32_t> continuation;
      for (std::uint32_t token; suffix >> token;) {
        continuation.push_back(token);
        state->Evaluate(token);
        write();
      }
      const std::vector<float> expected(state->Logits().begin(),
                                        state->Logits().end());
      state->Restore(*saved);
      for (const auto token : continuation)
        state->Evaluate(token);
      if (!std::equal(expected.begin(), expected.end(),
                      state->Logits().begin()))
        throw std::runtime_error("image snapshot continuation differs");
      auto fork = model->CreateSession();
      fork->Restore(*saved);
      for (const auto token : continuation)
        fork->Evaluate(token);
      if (fork->Save()->Digest() != state->Save()->Digest())
        throw std::runtime_error("image fork KV/hidden/logits differ");
      int cancellation_calls = 0;
      state->SetCancellationCheck([&] { return ++cancellation_calls == 20; });
      bool cancelled = false;
      try {
        state->EvaluateImage(embeddings);
      } catch (const std::runtime_error&) {
        cancelled = true;
      }
      if (!cancelled || state->Position() != 0)
        throw std::runtime_error("cancelled image block retained frontier");
      state->SetCancellationCheck({});
      state->Restore(*saved);
      for (const auto token : continuation)
        state->Evaluate(token);
      if (fork->Save()->Digest() != state->Save()->Digest())
        throw std::runtime_error("image cancellation recovery differs");
      if (!output)
        throw std::runtime_error("image logits write failed");
      std::cout << "PASS image snapshot/fork/cancellation continuation, rows="
                << continuation.size() + 1 << " position=" << state->Position()
                << std::endl;
      return 0;
    }
    if (std::string(argv[2]) == "--vision") {
      if (argc != 6)
        throw std::invalid_argument(
            "usage: MODEL --vision MMPROJ IMAGE DIRECTORY");
      model->LoadVision(argv[3]);
      const auto image =
          model->EncodeImage(gufo::core::ReadImageFile(argv[4]), argv[5]);
      std::ofstream dims(std::filesystem::path(argv[5]) / "dimensions.txt");
      dims << image.pixels.width << ' ' << image.pixels.height << '\n';
      std::ofstream normalized(std::filesystem::path(argv[5]) / "pixels.f32",
                               std::ios::binary);
      for (const auto byte : image.pixels.pixels) {
        const float value = float(byte) / 255.0F;
        normalized.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
      std::cout << "PASS finite vision embeddings, tokens=" << image.Tokens()
                << " weights=" << model->WeightBytes()
                << " scratch=" << model->ScratchBytes() << std::endl;
      return 0;
    }
    if (std::string(argv[2]) == "--assistant") {
      if (argc != 6)
        throw std::invalid_argument(
            "usage: MODEL --assistant ASSISTANT TOKENS DIRECTORY");
      model->LoadAssistant(argv[3]);
      auto state = model->CreateSession();
      std::ifstream input(argv[4]);
      for (std::uint32_t token; input >> token;)
        state->Evaluate(token);
      const std::vector<float> logits(state->Logits().begin(),
                                      state->Logits().end());
      const std::uint32_t seed =
          std::max_element(logits.begin(), logits.end()) - logits.begin();
      state->SetAssistantDirectory(argv[5]);
      const auto position = state->Position();
      const auto before = state->Save()->Digest();
      const auto proposals = state->Draft(seed, 7);
      if (before != state->Save()->Digest())
        throw std::runtime_error("assistant changed target KV/hidden/logits");
      state->SetAssistantDirectory({});
      int cancellation_calls = 0;
      state->SetCancellationCheck([&] { return ++cancellation_calls == 3; });
      bool cancelled = false;
      try {
        (void)state->Draft(seed, 7);
      } catch (const std::runtime_error&) {
        cancelled = true;
      }
      state->SetCancellationCheck({});
      if (!cancelled || before != state->Save()->Digest() ||
          proposals != state->Draft(seed, 7))
        throw std::runtime_error("assistant cancellation mutated target state");
      if (state->Position() != position ||
          !std::equal(logits.begin(), logits.end(), state->Logits().begin()))
        throw std::runtime_error("draft changed target frontier");
      std::ofstream output(std::filesystem::path(argv[5]) / "tokens.txt");
      output << seed << '\n';
      for (const auto token : proposals)
        output << token << '\n';
      std::cout << "PASS frozen target, drafts=" << proposals.size()
                << std::endl;
      return 0;
    }
    const bool multimodal = std::string(argv[2]) == "--multimodal-speculative";
    if (std::string(argv[2]) == "--speculative" || multimodal) {
      if (argc < 5)
        throw std::invalid_argument(
            "usage: MODEL --speculative ASSISTANT TOKENS [PREFIX_LENGTH]");
      model->LoadAssistant(argv[3]);
      std::vector<std::uint32_t> prompt;
      std::shared_ptr<const gufo::server::TextPromptContext> context;
      if (multimodal) {
        if (argc != 6)
          throw std::invalid_argument(
              "usage: MODEL --multimodal-speculative ASSISTANT MMPROJ IMAGE");
        model->LoadVision(argv[4]);
        gufo::server::ChatRequest request;
        request.reasoning.enabled = false;
        request.messages.emplace_back(
            gufo::models::common::ChatRole::kUser,
            "Describe the colors and shapes in this image.");
        request.messages.back().images.push_back(
            {0, std::make_shared<const std::vector<std::uint8_t>>(
                    gufo::core::ReadImageFile(argv[5]))});
        auto prepared =
            gufo::models::gemma4::CreateTextRunner(model)->PreparePrompt(
                request);
        prompt = std::move(prepared->tokens);
        context = std::move(prepared->context);
      } else {
        std::ifstream input(argv[4]);
        for (std::uint32_t token; input >> token;)
          prompt.push_back(token);
      }
      if (prompt.empty())
        throw std::runtime_error("empty speculation fixture");
      if (!multimodal && argc > 5) {
        const auto pattern = prompt;
        const auto size = std::stoul(argv[5]);
        prompt.resize(size);
        for (std::size_t i = 0; i < size; ++i)
          prompt[i] = pattern[i % pattern.size()];
      }
      auto plain = gufo::models::gemma4::CreateTextRunner(model, 0);
      auto speculative = gufo::models::gemma4::CreateTextRunner(model, 7);
      auto prefix = plain->CreateState();
      plain->SetPromptContext(*prefix, context);
      for (std::size_t offset = 0; offset < prompt.size();) {
        auto end = std::min(offset + 128, prompt.size());
        if (context)
          end = context->PrefillBoundary(end);
        offset += plain->Prefill(*prefix, prompt, offset, end - offset)
                      .consumed_tokens;
      }
      auto saved = plain->Snapshot(*prefix);
      std::size_t proposals = 0, accepted = 0;
      for (const auto config :
           {gufo::sampling::SamplingConfig{.seed = 42},
            gufo::sampling::SamplingConfig{.temperature = 0.8F,
                                           .top_k = 64,
                                           .top_p = 0.95F,
                                           .seed = 42,
                                           .repeat_penalty = 1.05F,
                                           .frequency_penalty = 0.1F}}) {
        auto single = plain->CreateState(), multi = speculative->CreateState();
        plain->RestoreOrFork(*single, *saved);
        speculative->RestoreOrFork(*multi, *saved);
        gufo::sampling::SamplerState one(config, prompt), many(config, prompt);
        std::vector<std::uint32_t> left, right;
        const auto generate = [&](const auto& runner, auto& state,
                                  auto& sampler, auto& ids, std::size_t width) {
          while (ids.size() < 24) {
            const auto step = runner.DecodeStep(
                state, std::min(width, 24 - ids.size()), sampler);
            if (step.failure)
              std::rethrow_exception(step.failure);
            proposals += step.draft_tokens;
            accepted += step.draft_accepted_tokens;
            for (const auto& selection : step.selections) {
              ids.push_back(selection.token);
              sampler.Accept(selection.token);
            }
            if (step.stop)
              break;
            if (step.selections.empty())
              throw std::runtime_error("speculation made no progress");
          }
        };
        generate(*plain, *single, one, left, 1);
        generate(*speculative, *multi, many, right, 8);
        if (left != right || one.rng_state() != many.rng_state() ||
            plain->CheckpointPosition(*single) !=
                speculative->CheckpointPosition(*multi))
          throw std::runtime_error("speculative output/RNG/position differs");
        auto a = plain->Snapshot(*single), b = speculative->Snapshot(*multi);
        const auto x = gufo::models::gemma4::SnapshotLogits(*a),
                   y = gufo::models::gemma4::SnapshotLogits(*b);
        if (!std::equal(x.begin(), x.end(), y.begin()) ||
            gufo::models::gemma4::SnapshotDigest(*a) !=
                gufo::models::gemma4::SnapshotDigest(*b))
          throw std::runtime_error(
              "speculative target KV/hidden/logits differ");
        std::cout << "PASS " << (config.temperature == 0 ? "greedy" : "sampled")
                  << " speculation: " << left.size()
                  << " tokens, full logits and RNG exact" << std::endl;
      }
      // Force each stop token through the request sampler without relying on
      // a model-generated stop in this short fixture.
      for (const auto eos : {1U, 106U, 50U}) {
        auto stopped = speculative->CreateState();
        speculative->RestoreOrFork(*stopped, *saved);
        const auto before = gufo::models::gemma4::SnapshotDigest(*saved);
        gufo::sampling::SamplerState sampler({.seed = 42}, prompt);
        sampler.DeferSample(eos);
        const auto result = speculative->DecodeStep(*stopped, 8, sampler);
        auto after = speculative->Snapshot(*stopped);
        if (!result.stop || !result.selections.empty() || result.failure ||
            before != gufo::models::gemma4::SnapshotDigest(*after))
          throw std::runtime_error("EOS mutated speculative target state");
      }
      auto cancelled = speculative->CreateState();
      speculative->RestoreOrFork(*cancelled, *saved);
      int cancellation_calls = 0;
      cancelled->SetCancellationCheck(
          [&] { return ++cancellation_calls == 3; });
      gufo::sampling::SamplerState sampler({.seed = 42}, prompt);
      bool failed = false;
      try {
        const auto result = speculative->DecodeStep(*cancelled, 8, sampler);
        failed = bool(result.failure);
      } catch (const std::runtime_error&) {
        failed = true;
      }
      if (!failed)
        throw std::runtime_error("MTP cancellation was ignored");
      cancelled->Invalidate();
      cancelled->SetCancellationCheck({});
      speculative->RestoreOrFork(*cancelled, *saved);
      if (gufo::models::gemma4::SnapshotDigest(*saved) !=
          gufo::models::gemma4::SnapshotDigest(
              *speculative->Snapshot(*cancelled)))
        throw std::runtime_error("MTP cancellation recovery differs");
      // Every draft performs at most 35 cancellation checks. Check 56 is
      // therefore inside target verification, including shortened drafts.
      speculative->RestoreOrFork(*cancelled, *saved);
      cancellation_calls = 0;
      cancelled->SetCancellationCheck(
          [&] { return ++cancellation_calls == 56; });
      gufo::sampling::SamplerState target_cancel_sampler({.seed = 42}, prompt);
      const auto target_cancel =
          speculative->DecodeStep(*cancelled, 8, target_cancel_sampler);
      if (!target_cancel.failure ||
          speculative->CheckpointPosition(*cancelled) != 0)
        throw std::runtime_error(
            "cancelled target verification retained frontier");
      cancelled->SetCancellationCheck({});
      speculative->RestoreOrFork(*cancelled, *saved);
      if (gufo::models::gemma4::SnapshotDigest(*saved) !=
          gufo::models::gemma4::SnapshotDigest(
              *speculative->Snapshot(*cancelled)))
        throw std::runtime_error(
            "target verification cancellation recovery differs");
      std::cout << "PASS all stop IDs, draft/target cancellation and restore"
                << std::endl;
      // Exercise natural stop after a partially accepted MTP block.
      const std::array stop_messages{gufo::models::common::ChatMessage{
          gufo::models::common::ChatRole::kUser,
          "What is 2+2? Answer only the digit."}};
      const auto stop_prompt = model->tokenizer().Encode(
          gufo::models::gemma4::RenderChat(stop_messages, {.enabled = false}),
          false);
      auto stop_prefix = plain->CreateState();
      for (std::size_t offset = 0; offset < stop_prompt.size();)
        offset += plain->Prefill(*stop_prefix, stop_prompt, offset, 128)
                      .consumed_tokens;
      auto stop_saved = plain->Snapshot(*stop_prefix);
      std::vector<std::uint32_t> stop_baseline;
      std::string stop_digest;
      for (const bool use_mtp : {false, true}) {
        const auto& runner = use_mtp ? *speculative : *plain;
        auto stopped = runner.CreateState();
        runner.RestoreOrFork(*stopped, *stop_saved);
        gufo::sampling::SamplerState stop_sampler({.seed = 42}, stop_prompt);
        std::vector<std::uint32_t> ids;
        bool stop = false;
        while (ids.size() < 16 && !stop) {
          const auto result = runner.DecodeStep(
              *stopped, std::min<std::size_t>(use_mtp ? 8 : 1, 16 - ids.size()),
              stop_sampler);
          if (result.failure)
            std::rethrow_exception(result.failure);
          for (const auto& selection : result.selections) {
            ids.push_back(selection.token);
            stop_sampler.Accept(selection.token);
          }
          stop = result.stop;
        }
        if (!stop || ids.empty())
          throw std::runtime_error("natural stop fixture failed");
        const auto digest =
            gufo::models::gemma4::SnapshotDigest(*runner.Snapshot(*stopped));
        if (!use_mtp) {
          stop_baseline = ids;
          stop_digest = digest;
        } else if (ids != stop_baseline || digest != stop_digest)
          throw std::runtime_error("partial-block natural stop state differs");
      }
      std::cout << "PASS natural stop after committed tokens" << std::endl;
      std::cout << "drafts=" << proposals << " accepted=" << accepted
                << std::endl;
      return 0;
    }
    if (std::string(argv[2]) == "--batch") {
      if (argc != 5)
        throw std::invalid_argument("usage: MODEL --batch TOKENS LOGITS");
      std::ifstream input(argv[3]);
      std::vector<std::uint32_t> tokens;
      for (std::uint32_t id; input >> id;)
        tokens.push_back(id);
      if (tokens.empty())
        throw std::invalid_argument("empty batch fixture");
      std::ofstream output(argv[4], std::ios::binary);
      auto state = model->CreateSession();
      std::size_t rows = 0;
      state->SetLogitObserver(
          [&](std::size_t position, std::span<const float> row) {
            if (position != ++rows)
              throw std::runtime_error("non-contiguous batch logits");
            output.write(reinterpret_cast<const char*>(row.data()),
                         row.size_bytes());
          });
      constexpr std::array<std::size_t, 6> widths{1, 8, 9, 32, 33, 128};
      for (std::size_t offset = 0, batch = 0; offset < tokens.size(); ++batch) {
        const auto count =
            std::min(widths[batch % widths.size()], tokens.size() - offset);
        state->EvaluateTokens(std::span(tokens).subspan(offset, count));
        offset += count;
      }
      if (!output || rows != tokens.size())
        throw std::runtime_error("batch output incomplete");
      std::cout << "PASS batch shapes 1/8/9/32/33/128, rows=" << rows
                << std::endl;
      return 0;
    }
    if (argc < 4)
      throw std::invalid_argument("missing logit output");
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3], std::ios::binary);
    auto session = model->CreateSession();
    if (argc > 4)
      session->SetLayerDirectory(argv[4]);
    for (std::uint32_t token; input >> token;) {
      session->Evaluate(token);
      const auto row = session->Logits();
      output.write(reinterpret_cast<const char*>(row.data()), row.size_bytes());
      if (session->Position() % 64 == 0)
        std::cout << "position=" << session->Position() << '\n';
    }
    if (!output || session->Position() == 0)
      throw std::runtime_error("invalid numerical fixture/output");
    std::cout << "rows=" << session->Position() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
