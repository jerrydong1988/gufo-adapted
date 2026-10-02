#ifndef GUFO_MODELS_QWEN_SERVE_PROMPT_HPP_
#define GUFO_MODELS_QWEN_SERVE_PROMPT_HPP_

// Shared Qwen-family prompt preparation for TextModelRunner adapters.
//
// Both the Qwen and Qwen3.8-Flash-Next runners render prompts through the
// compiled Qwen3.8 chat template and the Qwen vision stack; this header owns
// that shared logic so it lives with the Qwen model instead of serve.

#include <cstdint>
#include <memory>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {
struct ChatRequest;
struct TextPreparedPrompt;
struct TextPromptContext;
}  // namespace gufo::server

namespace gufo::models::qwen::vision {
class Encoder;
struct Prompt;
}  // namespace gufo::models::qwen::vision

namespace gufo::models::qwen::serve {

[[nodiscard]] tokenization::ChatTemplateOptions QwenChatOptions(
    const server::ChatRequest& request, std::uint32_t max_context);

[[nodiscard]] server::TextPreparedPrompt PrepareQwenPrompt(
    const server::ChatRequest& request,
    const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<vision::Encoder>& encoder, std::uint32_t max_context);

[[nodiscard]] std::shared_ptr<const vision::Prompt> QwenPrompt(
    const std::shared_ptr<const server::TextPromptContext>& context);

}  // namespace gufo::models::qwen::serve

#endif  // GUFO_MODELS_QWEN_SERVE_PROMPT_HPP_
