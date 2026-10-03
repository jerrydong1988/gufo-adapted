#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include "src/core/json.hpp"
#include "src/models/gemma4/model.hpp"

int main(int argc, char** argv) {
  using namespace gufo;
  using namespace models::gemma4;
  std::unique_ptr<core::GgufReader> reader;
  std::unique_ptr<Tokenizer> tokenizer;
  if (argc > 1) {
    std::string error;
    reader = core::GgufReader::OpenFile(argv[1], &error);
    if (!reader)
      throw std::runtime_error(error);
    const auto config = Config::FromGguf(*reader);
    config.ValidateWeights(*reader);
    if (!ValidateTemplate(*reader, &error))
      throw std::runtime_error(error);
    tokenizer = std::make_unique<Tokenizer>(*reader);
  }
  std::size_t checked = 0;
  for (const char* name : {"protocol-conventional.json", "protocol-qat.json"}) {
    if (tokenizer &&
        (std::string(name).find("qat") != std::string::npos) !=
            (std::string(argv[1]).find("qat") != std::string::npos))
      continue;
    std::ifstream input(std::filesystem::path(GEMMA4_FIXTURES) / name);
    const auto golden =
        json::parse(std::string(std::istreambuf_iterator<char>(input), {}));
    for (const auto& item : golden.find("chat")->items()) {
      std::vector<models::common::ChatMessage> messages;
      for (const auto& message : item.find("messages")->items()) {
        const auto role = message.member_str("role");
        messages.emplace_back(
            role == "system"      ? models::common::ChatRole::kSystem
            : role == "developer" ? models::common::ChatRole::kDeveloper
            : role == "assistant" ? models::common::ChatRole::kAssistant
            : role == "tool"      ? models::common::ChatRole::kTool
                                  : models::common::ChatRole::kUser,
            message.member_str("content"), message.member_str("name"),
            message.member_str("reasoning",
                               message.member_str("reasoning_content")));
        auto& value = messages.back();
        value.tool_call_id = message.member_str("tool_call_id");
        if (const auto* parts = message.find("content");
            parts && parts->is_array())
          for (const auto& part : parts->items()) {
            if (part.member_str("type") == "text")
              value.content += part.member_str("text");
            else
              value.images.push_back(
                  {value.content.size(),
                   std::make_shared<const std::vector<std::uint8_t>>(1, 1)});
          }
        if (const auto* calls = message.find("tool_calls"))
          for (const auto& call : calls->items()) {
            const auto& function = *call.find("function");
            models::common::ChatMessage::ToolCall converted;
            converted.id = call.member_str("id");
            converted.name = function.member_str("name");
            for (const auto& [key, argument] :
                 function.find("arguments")->members())
              converted.arguments.push_back(
                  {key, argument.is_string() ? argument.str() : argument.dump(),
                   argument.is_string()});
            value.tool_calls.push_back(std::move(converted));
          }
      }
      const auto rendered =
          RenderChat(messages, {.enabled = item.find("enabled")->as_bool()});
      if (rendered != item.member_str("rendered"))
        throw std::runtime_error("Gemma 4 template golden differs at case " +
                                 std::to_string(checked) +
                                 "\nexpected: " + item.member_str("rendered") +
                                 "\nactual: " + rendered);
      if (tokenizer) {
        std::vector<std::uint32_t> expected;
        for (const auto& token : item.find("tokens")->items())
          expected.push_back(token.as_size());
        if (tokenizer->Encode(rendered, false) != expected)
          throw std::runtime_error("Gemma 4 rendered tokens differ");
      }
      ++checked;
    }
    if (tokenizer) {
      for (const auto& item : golden.find("tokenizer")->items()) {
        std::vector<std::uint32_t> expected;
        for (const auto& token : item.find("tokens")->items())
          expected.push_back(token.as_size());
        if (tokenizer->Encode(item.member_str("text"),
                              item.find("add_special")->as_bool()) != expected)
          throw std::runtime_error("Gemma 4 tokenizer golden differs: " +
                                   item.member_str("text"));
        ++checked;
      }
    }
  }
  assert(Config{}.KvBytes(4096) == 2348810240ULL);
  if (argc > 2) {
    std::string error;
    auto assistant = core::GgufReader::OpenFile(argv[2], &error);
    if (!assistant)
      throw std::runtime_error(error);
    bool rejected = false;
    try {
      ValidateAssistant(*assistant, *reader);
    } catch (const std::exception&) {
      rejected = true;
    }
    if (rejected != (argc > 3 && std::string(argv[3]) == "--reject-assistant"))
      throw std::runtime_error("assistant compatibility result differs");
    std::cout << "PASS: assistant compatibility "
              << (rejected ? "rejected" : "accepted") << '\n';
  }
  std::cout << "PASS: " << checked
            << " independent Gemma 4 protocol fixtures\n";
}
