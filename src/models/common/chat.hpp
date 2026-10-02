#ifndef GUFO_MODELS_COMMON_CHAT_HPP_
#define GUFO_MODELS_COMMON_CHAT_HPP_

// Model-neutral conversation types shared by every text model package.
//
// These structs were originally defined for the Qwen chat template; they carry
// no Qwen-specific semantics (roles, tool calls, image parts), so they live
// here. New model packages include this header instead of reaching into
// another model's directory. The Qwen header keeps working through aliases.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::common {

/// Chat message roles supported by the shared serving boundary.
enum class ChatRole : std::uint8_t {
  kSystem = 0,
  kDeveloper = 1,
  kUser = 2,
  kAssistant = 3,
  kTool = 4,
};

[[nodiscard]] constexpr std::string_view ToString(ChatRole role) noexcept {
  switch (role) {
    case ChatRole::kSystem:
    case ChatRole::kDeveloper:
      return "system";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

/// A structured input message for chat formatting.
struct ChatMessage {
  ChatMessage() = default;
  ChatMessage(ChatRole message_role, std::string message_content,
              std::string message_name = {}, std::string message_thought = {})
      : role(message_role),
        content(std::move(message_content)),
        name(std::move(message_name)),
        thought(std::move(message_thought)) {}

  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string name;     ///< Optional function/tool name
  std::string thought;  ///< Optional thinking/reasoning prefix
  std::string tool_call_id;
  struct ImagePart {
    /// Insert an image before this byte of content. Equal offsets preserve
    /// input order; image-only messages use offset zero.
    std::size_t offset{0};
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
  };
  std::vector<ImagePart> images;

  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string id;
    std::string name;
    std::vector<ToolArgument> arguments;
  };

  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete validated HTTP tool object, preserving field order/extensions.
  std::string definition_json;
};

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_CHAT_HPP_
