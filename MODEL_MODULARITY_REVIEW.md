# Model modularity implementation review

Reviewed on 2026-10-02, on branch `experiment01` at commit `e8e033d`, against
baseline `c2ea1cf`.

The review found five reproducible bugs in the new generic CLI paths and common
validation harness. The moved model adapters appear behavior-preserving, and all
seven focused CPU tests passed. These findings concern cases the existing tests
do not cover.

This is a review report. Production fixes for all five findings landed in commit bc91e46; see its commit message for the per-finding summary and validation evidence.

## 1. Newly registered architectures fall through to Qwen in ordinary prompt

**Location:** [src/cli/prompt/prompt.cpp](src/cli/prompt/prompt.cpp), line 1220.

The registry correctly resolves a new model package, but the condition selecting
generic generation is:

```cpp
if (opt.generic || package_name == "qwen4exp") {
```

A newly registered architecture with a different package name therefore falls
through to `QwenGpuExecutor::CreateFromGguf` unless the user explicitly supplies
`--generic`. Registration alone does not provide the ordinary prompt path that
the integration contract promises.

**Reproduction:** A registered test package named `review-new-architecture`,
handling GGUF architecture `review-arch`, was passed to ordinary `prompt`.
The command returned exit code 1 with:

```text
Error creating Qwen GPU executor: Missing required Qwen tensor: token_embd.weight
REVIEW rc=1 loads=0 template_checks=0 states=0 prefills=0 selects=0
```

The same package generated successfully when `--generic` was supplied.

**Suggested fix:** Route packages without a specialized prompt implementation to
the generic path. Add a dispatch test using a package whose name is neither Qwen
nor DeepSeek.

## 2. Generic benchmarks can succeed without running a workload

**Location:** [src/cli/bench/bench.cpp](src/cli/bench/bench.cpp), lines 1500-1501;
the relevant option parsing is at lines 1378-1381.

The parser deliberately clears `n_gens` when only `-p` is supplied, and clears
`n_prompts` when only `-n` is supplied. The generic benchmark nests the generation
loop inside the prompt loop, so either empty list prevents every workload from
running.

**Reproduction:** Both of these argument combinations loaded the test package and
returned success without creating a state or executing inference:

```text
bench --generic -m <test.gguf> -p 64
bench --generic -m <test.gguf> -n 8

REVIEW rc=0 loads=1 template_checks=0 states=0 prefills=0 selects=0
```

Supplying both `-p 64 -n 8` exercised two generations, confirming that the test
package itself could execute.

**Suggested fix:** Handle prefill-only and generation-only workloads explicitly,
consistent with the existing CLI semantics. Test both forms and verify that a
successful benchmark actually executes the requested workload.

## 3. Generic benchmarks silently ignore validation and depth options

**Location:** [src/cli/bench/bench.cpp](src/cli/bench/bench.cpp),
`RunGenericBenchmark`, beginning at line 1435.

The generic path does not consume `logit_eval_path`, `logit_out`,
`validate_prefill_tokens`, or `n_depths`. It can return success after an ordinary
generation benchmark even though the user requested numerical validation or a
different context depth.

**Reproduction:** The test package was run with:

```text
bench --generic -m <test.gguf> -p 64 -n 8 \
  --logit-eval <nonexistent-corpus.txt> --logit-out <output-prefix>
```

The command returned exit code 0 and printed ordinary generic benchmark results.
It never attempted to read the nonexistent corpus. Its token determinism check
does not substitute for the requested logit evaluation.

**Suggested fix:** Reject unsupported validation and depth options before loading
the model, or implement their requested behavior. Keeping full-logit evaluation
model-specific is reasonable, but the generic path must report that limitation.

## 4. Generic prompting bypasses chat-template validation

**Location:** [src/cli/prompt/prompt.cpp](src/cli/prompt/prompt.cpp), line 379.

`RunGenericPrompt` calls `package.Load()` without calling
`package.ValidateTemplate()`. The package loaders do not perform that validation
internally. Consequently, the generic chat prompt path bypasses the compatibility
check used by serving and the existing Flash-Next prompt path.

**Reproduction:** The test package's `ValidateTemplate` implementation always
returns false with an unsupported-template error. Generic prompting nevertheless
loaded the package, generated four tokens, and returned success:

```text
REVIEW rc=0 loads=1 template_checks=0 states=1 prefills=1 selects=4
```

**Suggested fix:** Validate the artifact's template before loading and rendering
when chat templating is enabled. Test that a rejected template prevents both
model loading and generation. Preserve the intended raw-prompt behavior when
chat templating is disabled.

## 5. The common validation harness passes a decoder that makes no progress

**Location:**
[src/models/common/validate/validate.cpp](src/models/common/validate/validate.cpp),
lines 171-172 and 192-203.

Both decode-equivalence loops treat an empty selection list as a normal reason to
finish, even when `stop` is false. If both paths return this invalid result, the
empty token vectors compare equal and the check passes. The runner pool correctly
rejects this combination, but the validation harness calls the runner directly
and does not enforce that contract.

**Reproduction:** A deliberately broken test runner returned an empty
`TextDecodeStep`, with no failure and `stop=false`. The report included:

```text
decode_single_vs_multi_token passed=1 skipped=0 0 tokens identical on both decode paths
overall_passed=1
```

The test runner did not advertise snapshots, so the snapshot check was explicitly
skipped. This was sufficient for the overall report to pass despite invalid
decoding behavior.

**Suggested fix:** Fail on empty selections without a stop signal. Enforce the
other decode-step invariants, including the requested token budget and absence
of embedded stop selections, in the harness as well. Add a deliberately broken
runner fixture to verify that the harness detects these violations.

## Validation performed

The affected CPU test targets were checked with CMake before running CTest. All
seven focused tests passed:

- `openai_chat_test`
- `http_server_test`
- `text_model_runner_test`
- `text_generation_scheduler_test`
- `model_registry_test`
- `model_ops_test`
- `qwen38_flash_next.ops_parity`

The GPU-build CLI and package libraries were also built for a standalone review
reproducer. It used a minimal GGUF and a registered fake model package to exercise
the actual prompt, benchmark, and validation code without loading GPU weights.
The local reproducer is under the ignored `build/review-modularity/` directory;
it is not a committed repository test.

The three extracted model runner adapters were compared with their baseline
definitions. Their changes were predominantly namespace qualification and the
new output-dialect declarations; no inference arithmetic changes were identified
in those adapter moves.

## Limits

Real-model GPU matched-token logits, perplexity, performance, and Linux CI were
not independently rerun during this review. The implementation's existing
validation notes report additional model checks; this report does not treat
those notes as independently reproduced results.
