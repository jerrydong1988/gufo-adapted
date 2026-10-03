#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/gemma4/model.hpp"

namespace gufo::models::gemma4 {
namespace {

std::uint32_t Required(const core::GgufReader& reader, std::string_view key,
                       std::uint32_t expected) {
  const auto value = reader.GetMetadataUint32(key);
  if (!value || *value != expected) {
    throw std::invalid_argument("Gemma 4 31B requires " + std::string(key) +
                                " = " + std::to_string(expected));
  }
  return *value;
}

void RequiredFloat(const core::GgufReader& reader, std::string_view key,
                   float expected) {
  const auto value = reader.GetMetadataFloat32(key);
  if (!value || *value != expected) {
    throw std::invalid_argument("unsupported Gemma 4 metadata: " +
                                std::string(key));
  }
}

std::vector<std::uint64_t> UnsignedArray(const core::GgufReader& reader,
                                         std::string_view key) {
  const auto* value = reader.FindMetadata(key);
  if (value &&
      std::holds_alternative<std::vector<std::int64_t>>(value->value)) {
    const auto& signed_values =
        std::get<std::vector<std::int64_t>>(value->value);
    return {signed_values.begin(), signed_values.end()};
  }
  if (!value ||
      !std::holds_alternative<std::vector<std::uint64_t>>(value->value)) {
    throw std::invalid_argument("missing Gemma 4 array: " + std::string(key));
  }
  return std::get<std::vector<std::uint64_t>>(value->value);
}

std::string Pair(std::string_view left, std::string_view right) {
  std::string key(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

std::string Trim(std::string_view text) {
  const auto unicode =
      icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), text.size()));
  const auto whitespace = [](UChar32 c) {
    return u_isUWhiteSpace(c) || (c >= 0x1c && c <= 0x1f);
  };
  int32_t first = 0, end = unicode.length();
  while (first < end && whitespace(unicode.char32At(first)))
    first = unicode.moveIndex32(first, 1);
  while (end > first) {
    const auto previous = unicode.moveIndex32(end, -1);
    if (!whitespace(unicode.char32At(previous)))
      break;
    end = previous;
  }
  std::string result;
  unicode.tempSubStringBetween(first, end).toUTF8String(result);
  return result;
}

std::string StripThinking(std::string_view text) {
  std::string result;
  while (!text.empty()) {
    const auto end = text.find("<channel|>");
    const auto part = text.substr(0, end);
    result.append(part.substr(0, part.find("<|channel>")));
    if (end == std::string_view::npos)
      break;
    text.remove_prefix(end + 10);
  }
  return Trim(result);
}

}  // namespace

Config Config::FromGguf(const core::GgufReader& reader) {
  if (reader.GetMetadataString("general.architecture") != "gemma4") {
    throw std::invalid_argument("expected Gemma 4 target architecture");
  }
  Config config;
  Required(reader, "gemma4.block_count", config.layers);
  Required(reader, "gemma4.embedding_length", config.hidden);
  Required(reader, "gemma4.feed_forward_length", config.intermediate);
  Required(reader, "gemma4.attention.head_count", config.heads);
  Required(reader, "gemma4.attention.key_length", 512);
  Required(reader, "gemma4.attention.value_length", 512);
  Required(reader, "gemma4.attention.key_length_swa", 256);
  Required(reader, "gemma4.attention.value_length_swa", 256);
  Required(reader, "gemma4.rope.dimension_count", 512);
  Required(reader, "gemma4.rope.dimension_count_swa", 256);
  Required(reader, "gemma4.attention.sliding_window", config.window);
  Required(reader, "gemma4.attention.shared_kv_layers", 0);
  Required(reader, "gemma4.embedding_length_per_layer_input", 0);
  Required(reader, "gemma4.context_length", config.native_context);
  RequiredFloat(reader, "gemma4.attention.layer_norm_rms_epsilon",
                config.epsilon);
  RequiredFloat(reader, "gemma4.final_logit_softcapping", config.softcap);
  RequiredFloat(reader, "gemma4.rope.freq_base", config.global_rope_base);
  RequiredFloat(reader, "gemma4.rope.freq_base_swa", config.local_rope_base);
  const auto& heads = UnsignedArray(reader, "gemma4.attention.head_count_kv");
  const auto& local =
      UnsignedArray(reader, "gemma4.attention.sliding_window_pattern");
  if (heads.size() != config.layers || local.size() != config.layers) {
    throw std::invalid_argument(
        "Gemma 4 attention arrays must cover all layers");
  }
  for (std::size_t i = 0; i < config.layers; ++i) {
    if (heads[i] != config.KvHeads(i) || local[i] != config.Local(i)) {
      throw std::invalid_argument("unsupported Gemma 4 attention pattern");
    }
  }
  return config;
}

void Config::ValidateWeights(const core::GgufReader& reader) const {
  using core::GgmlType;
  std::unordered_set<std::string> expected;
  const auto check = [&](std::string name, std::vector<std::uint64_t> shape,
                         bool dense) {
    const auto* tensor = reader.FindTensor(name);
    if (!tensor || tensor->dimensions != shape) {
      throw std::invalid_argument("invalid Gemma 4 tensor shape: " + name);
    }
    const bool supported =
        tensor->type == GgmlType::kF32 || tensor->type == GgmlType::kQ4_0 ||
        tensor->type == GgmlType::kQ4_K || tensor->type == GgmlType::kQ5_K ||
        tensor->type == GgmlType::kQ6_K;
    if (!supported || (dense && tensor->type != GgmlType::kF32)) {
      throw std::invalid_argument("unsupported Gemma 4 tensor storage: " +
                                  name);
    }
    expected.insert(std::move(name));
  };
  check("token_embd.weight", {hidden, vocabulary}, false);
  check("output_norm.weight", {hidden}, true);
  check("rope_freqs.weight", {256}, true);
  for (std::size_t i = 0; i < layers; ++i) {
    const auto prefix = "blk." + std::to_string(i) + ".";
    const auto width = HeadWidth(i);
    check(prefix + "attn_q.weight", {hidden, width * heads}, false);
    check(prefix + "attn_k.weight", {hidden, width * KvHeads(i)}, false);
    if (Local(i))
      check(prefix + "attn_v.weight", {hidden, width * KvHeads(i)}, false);
    check(prefix + "attn_output.weight", {width * heads, hidden}, false);
    check(prefix + "attn_q_norm.weight", {width}, true);
    check(prefix + "attn_k_norm.weight", {width}, true);
    for (const auto* name :
         {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"}) {
      check(prefix + name + ".weight", {hidden}, true);
    }
    check(prefix + "ffn_gate.weight", {hidden, intermediate}, false);
    check(prefix + "ffn_up.weight", {hidden, intermediate}, false);
    check(prefix + "ffn_down.weight", {intermediate, hidden}, false);
    check(prefix + "layer_output_scale.weight", {1}, true);
  }
  for (const auto& tensor : reader.GetTensors()) {
    if (!expected.contains(std::string(tensor.name))) {
      throw std::invalid_argument("unsupported Gemma 4 tensor: " +
                                  std::string(tensor.name));
    }
  }
}

std::size_t Config::KvBytes(std::uint32_t context) const {
  std::size_t elements = 0;
  for (std::size_t i = 0; i < layers; ++i) {
    elements += std::size_t(HeadWidth(i)) * KvHeads(i) *
                (Local(i) ? std::min(window, context) : context) * 2;
  }
  return elements * sizeof(float);
}

void ValidateAssistant(const core::GgufReader& assistant,
                       const core::GgufReader& target) {
  if (assistant.GetMetadataString("general.architecture") != "gemma4-assistant")
    throw std::invalid_argument("expected gemma4-assistant sidecar");
  const auto required = [&](std::string_view name, std::uint32_t value) {
    Required(assistant, "gemma4-assistant." + std::string(name), value);
  };
  for (const auto& [name, value] :
       std::vector<std::pair<std::string_view, std::uint32_t>>{
           {"block_count", 4},
           {"embedding_length", 1024},
           {"embedding_length_out", 5376},
           {"feed_forward_length", 8192},
           {"nextn_predict_layers", 4},
           {"attention.head_count", 32},
           {"attention.key_length", 512},
           {"attention.value_length", 512},
           {"attention.key_length_swa", 256},
           {"attention.value_length_swa", 256},
           {"attention.sliding_window", 1024},
           {"attention.shared_kv_layers", 4},
           {"embedding_length_per_layer_input", 0},
           {"rope.dimension_count", 512},
           {"rope.dimension_count_swa", 256}})
    required(name, value);
  const auto context =
      assistant.GetMetadataUint32("gemma4-assistant.context_length");
  if (context != 131072 && context != 262144)
    throw std::invalid_argument("unsupported Gemma assistant context");
  RequiredFloat(assistant, "gemma4-assistant.attention.layer_norm_rms_epsilon",
                1e-6F);
  RequiredFloat(assistant, "gemma4-assistant.rope.freq_base", 1000000.0F);
  RequiredFloat(assistant, "gemma4-assistant.rope.freq_base_swa", 10000.0F);
  if (UnsignedArray(assistant, "gemma4-assistant.attention.head_count_kv") !=
          std::vector<std::uint64_t>{16, 16, 16, 4} ||
      UnsignedArray(assistant,
                    "gemma4-assistant.attention.sliding_window_pattern") !=
          std::vector<std::uint64_t>{1, 1, 1, 0})
    throw std::invalid_argument(
        "unsupported Gemma assistant attention pattern");
  // The supplied assistants omit tokenizer metadata: their output IDs index
  // the target's 262144-entry vocabulary. Keep the two published sets distinct.
  const auto* embedding = target.FindTensor("token_embd.weight");
  if (!embedding)
    throw std::invalid_argument("Gemma assistant needs target embeddings");
  const auto storage = embedding->type == core::GgmlType::kQ4_0
                           ? core::GgmlType::kQ4_0
                           : core::GgmlType::kQ8_0;
  std::unordered_set<std::string> expected;
  const auto check = [&](std::string name, std::vector<std::uint64_t> shape,
                         bool dense) {
    const auto* tensor = assistant.FindTensor(name);
    if (!tensor || tensor->dimensions != shape ||
        tensor->type != (dense ? core::GgmlType::kF32 : storage))
      throw std::invalid_argument("incompatible Gemma assistant tensor: " +
                                  name);
    expected.insert(std::move(name));
  };
  check("token_embd.weight", {1024, 262144}, false);
  check("output_norm.weight", {1024}, true);
  check("rope_freqs.weight", {256}, true);
  check("nextn.pre_projection.weight", {10752, 1024}, false);
  check("nextn.post_projection.weight", {1024, 5376}, false);
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const std::uint64_t width = i == 3 ? 512 : 256;
    check(prefix + "attn_q.weight", {1024, 32 * width}, false);
    check(prefix + "attn_output.weight", {32 * width, 1024}, false);
    check(prefix + "attn_q_norm.weight", {width}, true);
    for (const auto* name :
         {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"})
      check(prefix + name + ".weight", {1024}, true);
    check(prefix + "ffn_gate.weight", {1024, 8192}, false);
    check(prefix + "ffn_up.weight", {1024, 8192}, false);
    check(prefix + "ffn_down.weight", {8192, 1024}, false);
    check(prefix + "layer_output_scale.weight", {1}, true);
  }
  if (assistant.GetTensors().size() != expected.size())
    throw std::invalid_argument("unexpected Gemma assistant tensors");
}

Tokenizer::Tokenizer(const core::GgufReader& reader) {
  if (reader.GetMetadataString("tokenizer.ggml.model") != "gemma4") {
    throw std::invalid_argument("Gemma 4 requires its embedded BPE tokenizer");
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto& types = UnsignedArray(reader, "tokenizer.ggml.token_type");
  if (tokens.size() != 262144 || types.size() != tokens.size()) {
    throw std::invalid_argument("invalid Gemma 4 vocabulary");
  }
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens_.emplace_back(tokens[i]);
    token_ids_.emplace(tokens_[i], i);
    if (types[i] != 1 && types[i] != 6)
      special_.emplace_back(tokens_[i], i);
  }
  std::ranges::sort(special_, [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (merges.empty())
    throw std::invalid_argument("missing Gemma 4 BPE merges");
  for (std::size_t i = 0; i < merges.size(); ++i) {
    const auto split = merges[i].find(' ', 1);
    if (split == std::string_view::npos)
      throw std::invalid_argument("invalid Gemma 4 BPE merge");
    merges_.emplace(
        Pair(merges[i].substr(0, split), merges[i].substr(split + 1)), i);
  }
  for (std::size_t i = 0; i < bytes_.size(); ++i) {
    char token[8];
    std::snprintf(token, sizeof(token), "<0x%02X>", static_cast<unsigned>(i));
    const auto found = token_ids_.find(token);
    if (found == token_ids_.end())
      throw std::invalid_argument("missing Gemma 4 byte fallback");
    bytes_[i] = found->second;
  }
  // The pinned reference corrects legacy Gemma 4 conversions that set this
  // false. Gemma's raw-token API always inserts BOS when add_special is true.
  Required(reader, "tokenizer.ggml.bos_token_id", 2);
  add_bos_ = true;
}

void Tokenizer::EncodePlain(std::string_view text,
                            std::vector<std::uint32_t>& ids) const {
  std::string normalized;
  for (const char c : text)
    normalized.append(c == ' ' ? "\xe2\x96\x81" : std::string(1, c));
  struct Symbol {
    std::size_t offset, length;
    int previous, next;
  };
  struct Candidate {
    std::uint32_t rank;
    int left, right;
    std::size_t length;
    bool operator<(const Candidate& other) const {
      return rank != other.rank ? rank > other.rank : left > other.left;
    }
  };
  std::size_t start = 0;
  while (start < normalized.size()) {
    const bool newline = normalized[start] == '\n';
    std::size_t end = start + 1;
    while (end < normalized.size() && (normalized[end] == '\n') == newline)
      ++end;
    const std::string_view piece(normalized.data() + start, end - start);
    if (newline && token_ids_.contains(std::string(piece))) {
      ids.push_back(token_ids_.at(std::string(piece)));
      start = end;
      continue;
    }
    std::vector<Symbol> symbols;
    for (std::size_t offset = 0; offset < piece.size();) {
      const auto c = static_cast<unsigned char>(piece[offset]);
      const std::size_t width =
          std::min<std::size_t>(piece.size() - offset, c < 0x80   ? 1
                                                       : c < 0xe0 ? 2
                                                       : c < 0xf0 ? 3
                                                                  : 4);
      const int index = symbols.size();
      symbols.push_back({offset, width, index - 1,
                         offset + width < piece.size() ? index + 1 : -1});
      offset += width;
    }
    std::priority_queue<Candidate> queue;
    const auto add = [&](int left, int right) {
      if (left < 0 || right < 0)
        return;
      const auto& a = symbols[left];
      const auto& b = symbols[right];
      const auto found = merges_.find(Pair(piece.substr(a.offset, a.length),
                                           piece.substr(b.offset, b.length)));
      if (found != merges_.end())
        queue.push({found->second, left, right, a.length + b.length});
    };
    for (std::size_t i = 1; i < symbols.size(); ++i)
      add(i - 1, i);
    while (!queue.empty()) {
      const auto candidate = queue.top();
      queue.pop();
      auto& left = symbols[candidate.left];
      auto& right = symbols[candidate.right];
      if (!left.length || !right.length || left.next != candidate.right ||
          left.length + right.length != candidate.length)
        continue;
      left.length += right.length;
      right.length = 0;
      left.next = right.next;
      if (right.next >= 0)
        symbols[right.next].previous = candidate.left;
      add(left.previous, candidate.left);
      add(candidate.left, left.next);
    }
    for (const auto& symbol : symbols) {
      if (!symbol.length)
        continue;
      const auto value = piece.substr(symbol.offset, symbol.length);
      const auto found = token_ids_.find(std::string(value));
      if (found != token_ids_.end())
        ids.push_back(found->second);
      else
        for (const unsigned char c : value)
          ids.push_back(bytes_[c]);
    }
    start = end;
  }
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text,
                                             bool add_special) const {
  std::vector<std::uint32_t> ids;
  if (add_special && add_bos_)
    ids.push_back(2);
  while (!text.empty()) {
    std::size_t first = text.size();
    const std::pair<std::string, std::uint32_t>* match = nullptr;
    for (const auto& token : special_) {
      const auto position = text.find(token.first);
      if (position < first) {
        first = position;
        match = &token;
      }
    }
    EncodePlain(text.substr(0, first), ids);
    if (!match)
      break;
    ids.push_back(match->second);
    text.remove_prefix(first + match->first.size());
  }
  return ids;
}

std::string Tokenizer::Decode(std::span<const std::uint32_t> ids) const {
  std::string result;
  for (const auto id : ids) {
    if (id >= tokens_.size())
      throw std::out_of_range("Gemma 4 token ID");
    const auto& token = tokens_[id];
    if (token.size() == 6 && token.starts_with("<0x")) {
      result.push_back(
          static_cast<char>(std::stoul(token.substr(3, 2), nullptr, 16)));
    } else {
      std::size_t offset = 0;
      while (offset < token.size()) {
        const auto space = token.find("\xe2\x96\x81", offset);
        result.append(token.substr(
            offset, space == std::string::npos ? space : space - offset));
        if (space == std::string::npos)
          break;
        result.push_back(' ');
        offset = space + 3;
      }
    }
  }
  return result;
}

bool ValidateTemplate(const core::GgufReader& reader, std::string* error) {
  const auto value = reader.GetMetadataString("tokenizer.chat_template");
  if (value &&
      crypto::Sha256Hex(
          std::span(reinterpret_cast<const std::uint8_t*>(value->data()),
                    value->size())) ==
          "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b")
    return true;
  if (error)
    *error = "unsupported Gemma 4 chat template identity";
  return false;
}

namespace {
std::string Argument(const json::Value& value) {
  if (value.is_string())
    return "<|\"|>" + value.str() + "<|\"|>";
  if (value.is_array()) {
    std::string result = "[";
    for (const auto& item : value.items()) {
      if (result.size() > 1)
        result += ',';
      result += Argument(item);
    }
    return result + ']';
  }
  if (value.is_object()) {
    auto members = value.members();
    std::sort(members.begin(), members.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string result = "{";
    for (const auto& [key, item] : members) {
      if (result.size() > 1)
        result += ',';
      result += key + ':' + Argument(item);
    }
    return result + '}';
  }
  return value.dump();
}
}  // namespace

std::string RenderChat(std::span<const common::ChatMessage> messages,
                       const ReasoningOptions& reasoning,
                       std::vector<RenderedImage>* images) {
  using common::ChatRole;
  if (messages.empty())
    throw std::invalid_argument("Gemma 4 requires chat messages");
  if (images)
    images->clear();
  std::string text = "<bos>";
  const auto image = [&](const common::ChatMessage::ImagePart& part) {
    if (!part.bytes || part.bytes->empty())
      throw std::invalid_argument("empty Gemma image attachment");
    if (images)
      images->push_back({text.size(), part.bytes});
    text += "<|image|>";
  };
  const auto content = [&](const common::ChatMessage& message) {
    const bool assistant = message.role == ChatRole::kAssistant;
    if (message.images.empty()) {
      text +=
          assistant ? StripThinking(message.content) : Trim(message.content);
      return;
    }
    std::size_t cursor = 0;
    for (const auto& part : message.images) {
      if (part.offset < cursor || part.offset > message.content.size() ||
          (part.offset < message.content.size() &&
           (static_cast<unsigned char>(message.content[part.offset]) & 0xc0) ==
               0x80))
        throw std::invalid_argument("invalid Gemma image placement");
      const auto segment = std::string_view(message.content)
                               .substr(cursor, part.offset - cursor);
      text += assistant ? StripThinking(segment) : Trim(segment);
      image(part);
      cursor = part.offset;
    }
    const auto segment = std::string_view(message.content).substr(cursor);
    text += assistant ? StripThinking(segment) : Trim(segment);
  };
  const bool thinking = reasoning.enabled.value_or(false);
  std::size_t start = 0;
  if (thinking || messages[0].role == ChatRole::kSystem ||
      messages[0].role == ChatRole::kDeveloper) {
    text += "<|turn>system\n";
    if (thinking)
      text += "<|think|>\n";
    if (messages[0].role == ChatRole::kSystem ||
        messages[0].role == ChatRole::kDeveloper) {
      if (!messages[0].images.empty())
        throw std::invalid_argument("Gemma system images are not supported");
      text += Trim(messages[0].content);
      start = 1;
    }
    text += "<turn|>\n";
  }
  std::ptrdiff_t last_user = -1;
  for (std::size_t i = start; i < messages.size(); ++i)
    if (messages[i].role == ChatRole::kUser)
      last_user = i;
  bool previous_assistant = false;
  enum class Last { Text, Call, Response };
  Last last = Last::Text;
  for (std::size_t i = start; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == ChatRole::kTool)
      throw std::invalid_argument(
          "Gemma tool result requires a preceding matching call");
    last = Last::Text;
    const bool assistant = message.role == ChatRole::kAssistant;
    if (!assistant || !previous_assistant) {
      text += "<|turn>";
      text += assistant ? "model"
              : message.role == ChatRole::kDeveloper
                  ? "developer"
                  : common::ToString(message.role);
      text += '\n';
    }
    if (static_cast<std::ptrdiff_t>(i) > last_user && !message.thought.empty())
      text += "<|channel>thought\n" + message.thought + "\n<channel|>";
    if (!assistant && !message.tool_calls.empty())
      throw std::invalid_argument("Gemma tool calls require an assistant role");
    for (const auto& call : message.tool_calls) {
      text += "<|tool_call>call:" + call.name + '{';
      auto arguments = call.arguments;
      std::sort(arguments.begin(), arguments.end(),
                [](const auto& a, const auto& b) { return a.name < b.name; });
      bool comma = false;
      for (const auto& argument : arguments) {
        if (comma)
          text += ',';
        comma = true;
        text += argument.name + ':' +
                Argument(argument.is_string ? json::Value(argument.value)
                                            : json::parse(argument.value));
      }
      text += "}<tool_call|>";
      last = Last::Call;
    }
    std::size_t next = i + 1;
    if (!message.tool_calls.empty()) {
      while (next < messages.size() && messages[next].role == ChatRole::kTool) {
        const auto& response = messages[next++];
        std::string name = response.name.empty() ? "unknown" : response.name;
        bool matched = false;
        for (const auto& call : message.tool_calls)
          if (!call.id.empty() && call.id == response.tool_call_id) {
            name = call.name;
            matched = true;
          }
        if (!matched)
          throw std::invalid_argument(
              "Gemma tool result identity does not match its call");
        text += "<|tool_response>response:" + name +
                "{value:" + Argument(json::Value(response.content)) +
                "}<tool_response|>";
        // The embedded template concatenates result text, then emits images
        // outside the DSL string, preserving their input order.
        for (const auto& part : response.images)
          image(part);
        last = Last::Response;
      }
    }
    const auto content_start = text.size();
    content(message);
    const bool has_content = text.size() != content_start;
    const bool continues =
        assistant && next < messages.size() &&
        messages[next].role == ChatRole::kAssistant &&
        (message.tool_calls.empty() || last == Last::Response);
    if (last == Last::Call)
      text += "<|tool_response>";
    else if (!continues && !(last == Last::Response && !has_content &&
                             next == messages.size()))
      text += "<turn|>\n";
    previous_assistant = assistant;
    i = next - 1;
  }
  if (last == Last::Text) {
    text += "<|turn>model\n";
    if (!thinking)
      text += "<|channel>thought\n<channel|>";
  } else if (last == Last::Response && thinking)
    text += "<|channel>thought\n";
  return text;
}

}  // namespace gufo::models::gemma4
