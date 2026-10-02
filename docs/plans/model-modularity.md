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
- [ ] A8. GPU validation: rebuild `release` + `gpu-test`, run
  `openai_chat_test`, `http_server_test`, Flash-Next prompt/bench smoke
  with `experimental/` weights, and a matched-token logit comparison
  between baseline and refactored binaries (exact equality required).

## Phase B: reusable operations with replaceable fast paths

Status: not started.

- [ ] B1. `src/models/common/ops/`: CPU reference implementations with
  exact numerical contracts for embedding gather, RMSNorm, RoPE, softmax,
  SwiGLU/GELU, and residual accumulation. Each op: analytic unit tests,
  no HIP dependency.
- [ ] B2. `OpBackend` selection interface: shape/dtype dispatch between a
  correct general implementation and a specialized fast path, with a
  measurement hook. No production model migrates until its parity tests
  exist.
- [ ] B3. Migrate one Flash-Next CPU reference path onto shared ops to
  prove reuse; keep the HIP path untouched.
- [ ] B4. Document the "correct first, specialize after measurement"
  progression in `docs/ADDING_MODELS.md`.

## Phase C: state contract + validation harness

Status: not started.

- [ ] C1. Document the `TextModelRunner` state contract explicitly
  (append/share-prefix/restore, recurrent checkpoints, speculative
  commit/undo, multimodal identity) as the core state abstraction.
- [ ] C2. `src/models/common/validate/`: common harness driven through
  the package interface: tokenization round-trip, prompt-render golden,
  prefill/decode determinism, snapshot restore equality, streaming
  equivalence, perplexity/logit fixtures.
- [ ] C3. Wire Flash-Next into the harness; generalize `bench --logit-eval`
  beyond its current single-model implementation.
- [ ] C4. Update `docs/ADDING_MODELS.md` to the new work table: new sizes
  vs new arrangements vs new ops vs new bottlenecks.

## Working rules

- Small, independently reviewable commits; one focused change per commit.
- No numerics changes in Phase A; every refactor is behavior-preserving.
- Linux compatibility preserved; shared changes still need Linux CI.
- Record build identities, commands, and results for every GPU check.
