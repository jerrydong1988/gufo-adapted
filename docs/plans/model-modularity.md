# Model modularity implementation plan

Living plan for implementing `GUFO_MODEL_MODULARITY_IDEAS.md` on the
`experiment01` branch. Goal: reshape Gufo so adding model support is
substantially easier, without regressing gfx1151 performance or Linux
compatibility.

Baseline: `c2ea1cf` building in `build/release/gufo.exe`
(see `/tmp/gufo-baseline-build.log`). All refactors must preserve exact
numerics until an explicit, measured optimization says otherwise.

## Phase A: one registration exposes a model to serve (foundation)

Status: implementation complete; `check-pr` + GPU validation running.

- [x] A1. `gufo_text_runtime` static lib: move `continuation_cache.cpp`,
  `continuation_disk_store.cpp`, `text_model_runner.cpp`,
  `text_generation_scheduler.cpp` out of the `gufo` executable target into a
  linkable library. Update test targets that compile these TUs directly.
  No code changes; pure CMake repackaging.
- [x] A2. `src/models/common/chat.hpp`: move `ChatRole`, `ChatMessage`,
  `ChatTool` (and `ToString`) out of `src/models/qwen/chat_template.hpp`
  into model-neutral headers. Keep aliases in the Qwen header so existing
  includes keep working. New models must not include Qwen headers for chat
  types.
- [x] A3. `src/models/common/model_package.hpp` + `registry.hpp/cpp`:
  define the `ModelPackage` interface (name, GGUF architectures handled,
  template validation, load into a `TextModelRunner`) and a process-wide
  registry. The interface forward-declares serve types so `gufo_core`
  does not depend on serve.
- [x] A4. Move the three `TextModelRunner` adapters out of
  `src/cli/serve/inference_backend.cpp` into per-model files:
  `src/models/qwen/serve_runner.cpp`, `src/models/deepseek_v4_flash/serve_runner.cpp`,
  `src/models/qwen38_flash_next/serve_runner.cpp`. Behavior unchanged;
  the file shrinks to a thin dispatcher over the registry.
- [x] A5. `OutputDialect`: move `<think>`/tool-call parsing behind a
  model-declared dialect. Default preserves today's exact behavior
  (reasoning delimiters + Qwen/DSML auto-detect). `TextGenerationBackend`
  exposes the loaded model's dialect; `openai_chat.cpp` consumes it.
- [x] A6. Registry fallback hooks for `bench`/`prompt` so a newly
  registered package can serve all three CLIs without touching dispatch
  call sites. Legacy per-model bench harnesses stay until Phase C.
- [x] A7. CPU tests: registry unit tests, dialect parity tests
  (`openai_chat_test` fixtures must pass unchanged), format/docs checks.
- [x] A8. GPU validation: rebuild `release` + `gpu-test`, run
  `openai_chat_test`, `http_server_test`, Flash-Next prompt/bench smoke
  with `experimental/` weights, and a matched-token logit comparison
  between baseline and refactored binaries (exact equality required).

Phase A validation (2026-10-02, `1631d08`, TheRock 10.0.0, gfx1151):

- `check-pr`: 36/36 pass (cpu-test).
- `openai_chat_test`, `http_server_test`, `model_registry_test`: pass on
  both cpu-test and gpu-test presets.
- Prompt parity (`--speculative mtp`, greedy, 32 tokens): TokenTrace
  `sha256=3d023d32...` identical to the `c2ea1cf` baseline binary.
- Matched-token full logits (`bench --logit-eval`, 1666-token corpus,
  schedules `1:1,2,3,4,5,6,7,8`): all 1664 rows x 2 schedules hash-equal
  to baseline (`logit-eval.py` MATCH on both).
- Serve smoke via registry: buffered/streaming reasoning split, Qwen tool
  calls, and mmproj vision input all behave as baseline.
- `qwen_vision_serving_test` (Flash-Next + MTP + mmproj, typed-overload
  path): pass, including cancellation/resume exactness, incremental
  oracle with zero logit error, and disk image-identity replay.

## Phase B: reusable operations with replaceable fast paths

Status: foundation complete; GPU dispatcher deferred.

- [x] B1. `src/models/common/ops/`: canonical scalar CPU references with
  exact numerical contracts (RMSNorm, L2Norm, Sigmoid/SiLU/Softplus,
  SwiGLU, Softmax, NEOX RoPE) plus analytic tests. Known divergences
  (Qwen's float64-epsilon RMSNorm, divide-form SiLU) are documented as
  deliberate non-adoptions.
- [x] B3. Flash-Next CPU oracle forwards its pure-math operators to the
  shared implementations; bit-exact parity is pinned by
  `qwen38_flash_next.ops_parity` with recorded checksums. HIP untouched.
- [ ] B2. `OpBackend` selection interface: deferred until two models share
  a GPU kernel with parity + measurement proofs (see `ops/README.md`).
- [x] B4. The progression is documented in `src/models/common/ops/README.md`;
  `bench --generic` and `prompt --generic` (via `runner_generate`) give every
  registered package working benchmarks and smoke generation.

## Phase C: state contract + validation harness

Status: complete.

- [x] C1. The `TextModelRunner` state contract is documented on the class
  itself (`text_model_runner.hpp`): token-count positions, immutable exact
  snapshots, restore-means-restore (recurrent/sliding-window state comes
  back, not just the position), speculative commit/rollback inside the
  runner with byte-identical greedy output, multimodal identity in the
  prompt context, and honest capability advertisement.
- [x] C2. `src/models/common/validate/`: harness driven through the package
  interface — tokenize determinism, render-and-tokenize smoke,
  single-step vs multi-token decode equivalence, and snapshot/restore
  fidelity. Model failures report as named failed checks, not crashes.
- [x] C3. Flash-Next wired in via `qwen38_flash_next_validate_test`
  (weights-gated, skips without `--model`): all four checks pass on
  `experimental/` weights with MTP. `bench --logit-eval` stays
  model-specific by design — full-logit numerical truth belongs with the
  model, while the common harness covers behavioral fidelity.
- [x] C4. `docs/ADDING_MODELS.md` updated: Step 0 dispatch reality is now
  the registry, Step 8 is package registration, the checklist requires the
  adapter + dialect + generic paths + harness, and the novelty work table
  matches the ideas doc.

## Working rules

- Small, independently reviewable commits; one focused change per commit.
- No numerics changes in Phase A; every refactor is behavior-preserving.
- Linux compatibility preserved; shared changes still need Linux CI.
- Record build identities, commands, and results for every GPU check.
