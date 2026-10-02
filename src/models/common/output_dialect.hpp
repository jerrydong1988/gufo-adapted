#ifndef GUFO_MODELS_COMMON_OUTPUT_DIALECT_HPP_
#define GUFO_MODELS_COMMON_OUTPUT_DIALECT_HPP_

// Model-declared output dialect: the delimiters and tool-call protocols a
// model emits. The shared HTTP layer parses generations through the loaded
// model's dialect instead of hard-coding one family's conventions.
//
// The default preserves today's behavior exactly: Qwen-style <think>
// reasoning plus Qwen/DSML tool-call auto-detection. A model package
// declares its dialect in its TextModelRunner descriptor; new models with
// different conventions only change their own declaration.

#include <string>

namespace gufo::models::common {

struct OutputDialect {
  /// Reasoning span delimiters. Empty strings disable reasoning parsing:
  /// output is content from the first token.
  std::string think_start{"<think>"};
  std::string think_end{"</think>"};
  /// Accepted tool-call protocols. When both are set, the wire format is
  /// detected from the emitted markup, as before.
  bool qwen_tool_calls{true};
  bool dsml_tool_calls{true};
};

}  // namespace gufo::models::common

#endif  // GUFO_MODELS_COMMON_OUTPUT_DIALECT_HPP_
