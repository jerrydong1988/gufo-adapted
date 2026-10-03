# Gemma 4 implementation log

Started 2026-10-02 on `experiment01`, baseline `6004601`.
This follows the Gemma implementation sequence in
[GUFO_MODEL_MODULARITY_IDEAS.md](../../GUFO_MODEL_MODULARITY_IDEAS.md).

## Gates and scope

1. Artifact manifests and pinned independent reference: complete for both targets,
   assistants and projectors.
2. Bounded text inference through registration (`prompt`, `serve`, generic
   `bench`), independent full logits and perplexity: complete for both targets.
3. Window boundaries, cancellation, continuation, snapshots, restore, fork and
   the common harness: complete for both targets; no harness checks skipped.
4. Assistant/MTP compatibility, intermediate parity, greedy/sampled verification,
   stop, cancellation and exact target-state recovery: complete for both sets.
5. Vision intermediate and image-window parity and HTTP coverage: complete for
   both sets, including QAT image serving with MTP. CLI image/MTP and exact
   unchanged-scalar checks also pass. Bounded text batching is retained after
   numerical qualification and a 4.51064x QAT prefill improvement. A first
   cross-model FP32 GEMM candidate is selected but not extracted without adopter
   proofs. GPU timeline profiling and Linux execution are blocked by missing
   local tools; synchronized wall measurements and Linux CI wiring are recorded.

The first supported target is the dense 60-layer 31B architecture at 4K context,
one session. Local QAT and conventional targets have separate numerical
baselines. Presence of an assistant/projector does not qualify that capability.
Weights and large numerical dumps remain under ignored `experimental/`/`build/`.

## Initial inspection

- Checkout is clean; no Gufo or llama inference processes were running.
- QAT target: Q4_0/FP32; conventional target: Q4_K/Q5_K/Q6_K/FP32.
- Both projectors have BF16/FP32 storage. Assistants differ between sets.
- A clean local llama.cpp checkout at
  `d2c35f60288338fdaaef63c90fb5904d920d590a` implements both `gemma4` and
  `gemma4-assistant`; its reference build will be verified before use.
- Toolchain: native Windows, TheRock ROCm 10.0.0, gfx1151, CMake/Ninja,
  MSVC headers/SDK and manifest vcpkg dependencies.
- CodeGraph was consulted first. It did not surface the recently added package
  files when queried by path; direct reads cover those missing entries.

## Mathematical contract

The 31B target scales token embeddings by sqrt(5376), uses five local layers
then one global layer, a 1024-token inclusive causal window, 256/512 head widths,
16/4 KV heads, head RMSNorm on Q/K and unweighted RMSNorm on V, attention scale
1, tanh GELU gating, pre/post attention and FFN normalization, layer output
scales, tied embeddings, and final logit softcapping at 30. Global proportional
RoPE must honor the stored frequency factors. Converted GGUF normalization
weights must be used according to the converter contract.

References:
[Google config](https://huggingface.co/google/gemma-4-31B-it/blob/419b2efe421994fdfd3394e621983d4cc511cd4f/config.json),
[Unsloth QAT model](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF),
[assistant notes](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/blob/main/MTP/README.md),
[Google MTP overview](https://ai.google.dev/gemma/docs/mtp/overview).

## Validation record

The entries below are chronological; statements about pending work describe
the state at that entry. Current results are consolidated in the gates above,
[QUALITY.md](../models/gemma4/QUALITY.md),
[BENCHMARKS.md](../models/gemma4/BENCHMARKS.md), and
[EXPERIMENTS.md](../models/gemma4/EXPERIMENTS.md).

### Reference and protocol fixtures

`src/models/gemma4/fixtures/artifacts.json` records complete tensor inventories,
storage counts, tokenizer/template hashes, six file hashes, and hash-matched
publisher revisions. Conventional: `c1ac76e99d5513b141e8adde7288b85c3f9c32ec`;
QAT: `43cc1aeb31adf47ec06a854507ce552cd9862e6f`. Projectors are different files
despite identical sizes. The conventional assistant is Q8_0; QAT is Q4_0.

The actual reference binary identifies itself as build 357, revision
`1d13fa1c5d6ed9f8cfdc924744b204070c32466d`, Clang 23.0.0. Its Gemma target
and assistant sources are identical to the inspected checkout's `d2c35f6`.
The standalone `llama_reference.cpp` links only this independent llama/ggml build.
It uses ROCm, 4K context, FP32 K/V, attention scale 1, flash attention disabled,
one sequence, and no assistant/projector. Token fixtures use the pinned
library; template fixtures execute the artifact's exact Jinja template.

Each set has 32 tokenizer and 12 chat goldens, covering raw/chat BOS, special
tokens, Unicode, byte fallback, newline runs, whitespace, history, assistant
continuations and thinking channels. The pinned reference corrects the QAT
artifact's legacy `add_bos_token=false`; raw encoding with special tokens still
adds BOS. Chat rendering explicitly supplies BOS and encodes without adding it.

### Text bring-up

Native model-private HIP execution and registration compile on Windows. CPU
chat-template fixtures pass (24 checks, both sets). Unsupported tools, vision,
speculation, concurrency and persistent snapshots fail explicitly.

Before full-logit comparisons, limits were set to per-row RMSE <= 0.1, maximum
absolute logit difference <= 1, and relative perplexity difference <= 1% against
the same artifact/token sequence. These limits cover independently rounded
reductions, not different weights. Full rows and next-token loss are retained;
top-1 agreement is diagnostic, not the acceptance criterion.

The first 65-token conventional trial failed against llama's default quantized
GEMV: maximum row RMSE 0.689567 and maximum error 3.400123. Its GEMV also
quantizes activations to Q8_1. An attempted Q8_1 native implementation failed
qualification (RMSE 0.422175, maximum error 2.3586, relative PPL difference
2.5487%) and was removed. Neither failure justifies relaxing numerical limits.

The retained FP32-activation implementation was then compared with an
independent FP32-accumulation BLAS oracle at the same pinned llama revision.
`tools/reference_blas.py` rebuilds only the reference backend's BLAS dispatch
in an isolated directory; original source/build stay unchanged. Batches above
eight tokens avoid MMVQ, fusion is disabled, and the public graph callback
selects FP32 accumulation. These settings avoid llama's additional activation
quantization and default FP16 BLAS accumulation. This is an arithmetic-matched
correctness oracle, not the performance reference.

The 65-token conventional comparison passes the original limits: maximum row
RMSE 0.000169957, maximum error 0.000878155, top-1 agreement 65/65. Reference
PPL 254.446374 versus native 254.446464 (relative difference 0.0000355%).
Embedding/input normalization match exactly; first local/full layer outputs
and the final normalized activation match within small FP32 reduction error.
The 277-token corpus passes for both artifact sets (all 262144 logits at every
position). Conventional: maximum RMSE 0.00592638, maximum error 0.0473521,
PPL 186.521962 vs 186.522252, relative difference 0.0001555%. QAT: maximum RMSE
0.000568722, maximum error 0.00265312, PPL 334.354836 vs 334.355527, relative
difference 0.0002066%. Both have 277/277 top-1 agreement. The corpus is a small
regression fixture, not a broad quality benchmark.

Commands (paths are repository-relative):

```powershell
# Oracle uses the isolated BLAS DLL, fusion disabled, FP32 K/V and accumulation.
$env:GGML_CUDA_DISABLE_FUSION = '1'
build/gemma4/reference-blas/llama-reference.exe <target> build/gemma4/corpus-tokens.txt <reference.f32> 277 <layer-directory> --f32
build/gpu-test/gemma4_validate.exe <target> build/gemma4/corpus-tokens.txt <native.f32>
python src/models/gemma4/tools/compare.py <reference.f32> <native.f32> build/gemma4/corpus-tokens.txt --output <comparison.json>
```

### State and assistant bring-up

The QAT common harness passes 4/4: tokenize determinism, chat rendering,
single/wide decode equivalence, and snapshot restore. In-memory snapshots retain
valid physical ring rows, full-attention rows, the post-final-norm hidden vector,
and full logits. Capture uses a private stream without model scratch. Restore
rejects another model/context and replaces overwritten rows. Persistent disk
snapshots remain unsupported. Real-model window-boundary qualification is running.

Assistant implementation follows the pinned reference: four layers, 1024-wide
hidden state, target embeddings plus post-final-norm activations projected from
10752 to 1024, recurrent 5376-wide output projection, target KV from local layer
58/full layer 59, and the SAME query position for every draft in a block. Drafts
never write target KV. The supplied assistants omit tokenizer metadata, so output
IDs index the target vocabulary; shapes, storage and the two artifact sets are
checked explicitly.

The initial verifier evaluates only committed target tokens sequentially. Greedy
proposals are point-mass distributions: a target sample accepts a proposal iff
it matches, otherwise that target sample becomes the replacement. This preserves
exact target sampling, filters, penalties and seeded RNG order. It promises no
batch-verification speedup. Assistant numerical parity, greedy/sampled state
parity and measured acceptance are pending.


QAT window qualification passes exact full-logit restoration and continuation at
positions 1023, 1024, 1025 and 2049 after 2052 tokens of cache writes. Forked
states also match, snapshot payload accounting agrees, and cancellation during
layer 20 invalidates the frontier; a subsequent request matches fresh
recomputation exactly. Command: `gemma4_validate <QAT-target> --state
build/gemma4/corpus-tokens.txt`. The conventional/window independent reference
checks remain pending.

Production CLI (`gufo version 600460118c23-dirty`) loads Gemma by registration.
QAT raw prompt, chat prompt (thinking off), and generic bench pp16/tg8 pass. The
chat answer is `4`; the short deterministic benchmark reports roughly 4 tok/s.
These are integration/debug results, not retained performance benchmarks.

The conventional assistant's seven-step independent comparison passes (FP32
BLAS, same artifacts and frozen target prefix). First-step maximum error is
0.000162 across pre-projection, four layer outputs, recurrence projection and
full 262144 logits. Proposal IDs match. Initial oracle setup exposed two
harness issues, both corrected: Windows ownership of the manually allocated
token array, and creating the shared-KV assistant context BEFORE target prefill.
The latter otherwise clears shared KV during initialization. Non-finite
reference intermediates now fail explicitly.

QAT MTP verification passes 24-token greedy and seeded sampled output parity,
exact full target logits, positions and RNG (temperature 0.8, top-k 64, top-p
0.95, repetition 1.05, frequency penalty 0.1). This fixture accepts 39 of 45
proposals, covering accepted and rejected drafts. Window/stop/cancellation MTP
qualification and throughput comparisons remain pending.

HTTP buffered/streaming visible text, reasoning and usage agree for QAT with
thinking enabled and disabled; prefix reuse is exercised by the second request.
The negative tool fixture initially omitted `model` and hit request validation
instead of the package. It has been corrected to require the package's explicit
unsupported-tool error before qualifying that check.

Vision is bounded initially to 280 soft tokens/image, 4K combined context, one
session. RGB preprocessing aligns to 48 pixels, uses Pillow bicubic RGB8 with
centered black padding, and matches the reference's 70..280 token budget. BF16
encoder/projector weights widen losslessly to FP32; the patch convolution's
scaled-pixel FP16 boundary is retained. The native target consumes whole image
blocks with local bidirectional/full-layer causal masks. Local ring capacity
includes 280 extra rows to preserve the past while writing a complete block.
The QAT encoder comparison passes 41 named intermediate tensors, including all
27 layer outputs: maximum RMSE 0.000679861 and maximum absolute error 0.0458984
under the previously declared 0.01/0.1 limits. Resized normalized RGB pixels are
bit-identical. The independent language-model image block plus 16 following text
tokens passes full-logit comparisons (max RMSE 0.000647707, max error 0.00297022,
17/17 top-1 agreement, next-token PPL relative error 0.000101874). Restoring the
image frontier and replaying those tokens produces exact full logits.

Two reference arithmetic settings needed correction before qualification:
the supplied projector omits `clip.use_gelu`, selecting generic CLIP Quick GELU,
whereas [Google's pinned vision configuration](https://huggingface.co/google/gemma-4-31B-it/blob/419b2efe421994fdfd3394e621983d4cc511cd4f/config.json)
specifies `gelu_pytorch_tanh`; and this Windows reference fork's BF16 WMMA
dispatch ignores the FP32 precision request for batches >=512. The isolated
oracle selects GELU tanh and uses a reproducible dispatch-only overlay to honor
`GGML_PREC_F32` before that route. Weights and numerical operators are unchanged;
no Gufo math is linked. `reference_blas.py` records exact build commands and
generates the overlay outside the reference checkout. Normal reference
performance must use its original DLL, separately from this numerical oracle.

Serving image preparation remains package-owned and CPU-only; GPU encoding runs
under the runner's execution lock. Cache identity includes projector SHA-256,
RGB pixels, dimensions and image positions, and follows immutable request
context through replay. One shared interface extension is demonstrated:
`TextPromptContext::PrefillBoundary` rounds prefill/checkpoint endpoints out of
bidirectional image blocks. Gemma consumes at most 280 tokens per atomic image
work unit; existing text and Qwen contexts preserve their original budgets.
Generic CLI prompting now forwards image attachments through `PreparePrompt`.
Tool definitions/output tool calls remain rejected, while validated incoming
tool history can carry images for Responses replay. API qualification pending.


### Serving and bounded prefill qualification

The QAT HTTP vision suite now passes direct Chat images, Chat history replay,
images in Responses tool results, Responses replay and both streaming forms.
A shared Auto-state delimiter bug exposed by buffered Responses is fixed with
exhaustive chunk-width fixtures for Gemma and existing Qwen markers. The five
focused shared CPU suites pass after rebuilding their targets. Incoming tool
history is validated; tool definitions and generated tool calls remain rejected.

The QAT assistant also passes all 49 seven-step intermediate tensors (maximum
RMSE 0.0000333239, error 0.000167847). Target KV immutability, assistant
cancellation recovery and all stop IDs have been added to the real-model gates.

A release pp2048/tg128 baseline is running with the matching assistant resident
but a plain AR control. The model-private FP32 image batch body is generalized
to causal text batches of at most 128. Scalar decode math remains unchanged.
Local text rings reserve 128 extra rows so a complete batch cannot overwrite an
earlier query's visible past. Vision rings retain their 280 extra rows.
State ABI changes to v2; no persistent snapshots are advertised. Diagnostics
capture all batch-row logits for independent comparisons at 1/8/9/32/33/128
widths. Existing limits apply; this change is not yet qualified or retained.

The optimize-kernel workflow is applied, with initial BENCHMARKS/QUALITY/
EXPERIMENTS records under docs/models/gemma4. Windows GPU timeline profiling is
unavailable because this TheRock distribution has no rocprofv3. Wall timings are
recorded separately and will not be labeled GPU profiles.


Both batched-text targets pass all 277 full-vocabulary rows at awkward widths.
QAT: maximum RMSE 0.000321345, error 0.00174093, PPL relative error 1.68095e-6.
Conventional: RMSE 0.00563335, error 0.0451210, PPL relative error 1.99374e-6.
Top-1 agrees for all rows in both sets. The QAT production A/B improves pp2048
from 3.31577 to 14.9562 tok/s (4.51064x), with the same 128 greedy IDs and a
-0.35% scalar decode control change. MTP is measured at 2.84602 vs AR 3.00696
tok/s; it is explicit opt-in, with no speedup claim. Details and timed scope
are in docs/models/gemma4/BENCHMARKS.md. Window/image checks remain gating.


QAT independent window comparison passes all 11 selected full-logit frontiers
at positions 20, 1023..1028 and 2049..2052: max RMSE 0.00244638, max error
0.00936890, top-1 11/11, matched next-token PPL relative error 1.05288e-6.
Conventional native snapshot/fork/cancellation/fresh-recompute checks also pass
at both windows; its independent comparison is running. Thirteen rebuilt
Windows CPU suites pass, including Qwen tokenizer/template, Flash-Next config,
MTP sampling, prompt lookup and scalar-op parity, alongside API, scheduler and
cache lifecycle checks.


Conventional independent window frontiers also pass (11 rows, max RMSE
0.0133574, error 0.0531216, 11/11 top-1, PPL relative error 2.82743e-6).
Both vision encoders pass 41 independent intermediate tensors at the existing
0.01/0.1 limits. Encoder-layer outputs are identical between supplied sets;
projected embeddings differ, so file/projector identity remains significant.
The QAT image block beginning at position 1008 passes all 17 subsequent
full-logit rows across the first local-window boundary (RMSE 0.000395339,
error 0.00225878, 17/17 top-1, PPL relative error 5.66942e-5). Image snapshot,
fork and cancellation recovery compare complete KV/hidden/logit SHA-256 values
exactly. Conventional image-window qualification is running.

Local Ubuntu WSL exists but has no Nix, compiler or CMake; Linux execution is
not claimed. The Gemma protocol test is added to both the bounded Linux
check-pr target and the Windows CPU workflow, without downloading models.
Actual-GGUF protocol checks pass 66 goldens for each target, accept their own
assistants and reject both cross-set assistant pairings.


Conventional image-window independent comparison passes (17 rows, RMSE
0.00114066, error 0.0105993, 17/17 top-1, PPL relative error 0.000351929).
All image snapshot/fork/cancellation checks compare complete native state exactly.
Both assistants are requalified on the final target runtime: 49 tensors per set,
matching seed/proposal IDs, SHA-256 of all target KV/hidden/logits unchanged after
seven drafts and after cancellation, and identical proposals on retry.
A pp16/tg8 generic CLI control improves short-prompt prefill 1.70557x while
retaining decode throughput. Three CLI contract suites and the updated Gemma
protocol test pass after rebuilding.


### Final integration and reproducibility

Both final common-harness runs, with their matching assistant resident, pass
4/4 with no skips: tokenizer determinism, chat rendering, single/multi-token
decode and snapshot restoration. Both combined image/MTP fixtures pass 24 greedy
and 24 sampled selections, exact complete target-state/RNG/position comparisons,
all stop IDs, draft/target cancellation and natural partial-block stop. QAT
proposes/accepts 52/33; conventional 74/28. These counts are correctness-fixture
observations, not throughput measurements.

The final HTTP test adds same-position image replacement and two-image order.
QAT passes all initial cases. A conventional immediate tool continuation returns
one newline and then stops, including on a cold request; an explicit follow-up
correctly reads both shapes/colors from the retained tool-result image. This
observed difference is retained in the test report. The final suite checks both
the original continuation's streaming equality and a semantic follow-up/replay.
The first rerun exposed a missing test dependency, resolved by an isolated
`build/gemma4/http-env` with Pillow 12.0.0; existing GUI environments are untouched.
Python numerical/fixture tools use Python 3.14.6, NumPy 2.5.2 and Jinja2 3.1.6.

Small exact token inputs are now retained in `src/models/gemma4/fixtures`:
277-token corpus, 65-token assistant prefix, 16-token image suffix, image-loss
alignment and 11 one-based window frontier positions. Generated window histories,
large full-logit/intermediate dumps, executable baselines and HTTP logs remain
under ignored `build/gemma4/`.

Production Gemma code is 3082 lines across nine private C++/HIP/header files.
Shared changes consist of package/build registration; generic CLI image forwarding;
an atomic prompt endpoint contract for bidirectional image blocks and cache
checkpoints; and a reasoning-filter fix for a complete delimiter arriving with
content in one chunk. Existing text contexts retain identity endpoint rounding.
There is no Gemma architecture dispatch branch and no cross-model GPU extraction.

Final-route reproduction (replace paths with the matching conventional set to
repeat that qualification; run one GPU process at a time):

```powershell
$gemmaTarget = 'experimental/unsloth-gemma4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf'
$gemmaAssistant = 'experimental/unsloth-gemma4-31b-qat/mtp-gemma-4-31B-it.gguf'
$gemmaProjector = 'experimental/unsloth-gemma4-31b-qat/mmproj-BF16.gguf'
$gemmaFixtures = 'src/models/gemma4/fixtures'

./tools/windows/build.ps1 -Preset gpu-test -Target gemma4_validate -Jobs 4
build/gpu-test/gemma4_validate.exe $gemmaTarget --harness $gemmaAssistant
build/gpu-test/gemma4_validate.exe $gemmaTarget --batch "$gemmaFixtures/corpus-tokens.txt" build/gemma4/batch.f32
build/gpu-test/gemma4_validate.exe $gemmaTarget --state "$gemmaFixtures/corpus-tokens.txt" build/gemma4/window.f32
build/gpu-test/gemma4_validate.exe $gemmaTarget --assistant $gemmaAssistant "$gemmaFixtures/assistant-tokens.txt" build/gemma4/assistant-check
build/gpu-test/gemma4_validate.exe $gemmaTarget --speculative $gemmaAssistant "$gemmaFixtures/corpus-tokens.txt" 1023
build/gpu-test/gemma4_validate.exe $gemmaTarget --vision $gemmaProjector "$gemmaFixtures/shapes.png" build/gemma4/vision-check
build/gpu-test/gemma4_validate.exe $gemmaTarget --multimodal-speculative $gemmaAssistant $gemmaProjector "$gemmaFixtures/shapes.png"
build/gpu-test/gemma4_validate.exe $gemmaTarget --image-state $gemmaProjector "$gemmaFixtures/corpus-tokens.txt" build/gemma4/vision-check/projected.bin "$gemmaFixtures/image-suffix-tokens.txt" build/gemma4/image-window.f32 1008

./tools/windows/build.ps1 -Preset release -Target gemma4_bench -Jobs 4 -CMakeArgs '-DGUFO_BUILD_TOOLS=ON'
build/release/gemma4_bench.exe $gemmaTarget $gemmaAssistant build/gemma4/benchmark-repeat.json 2048

build/gemma4/http-env/Scripts/python.exe tests/models/gemma4/http_smoke.py --engine build/release/gufo.exe --model $gemmaTarget --mmproj $gemmaProjector --image "$gemmaFixtures/shapes.png" --output build/gemma4/http-repeat.json
# Add --mtp-model $gemmaAssistant to qualify the MTP serving route.
# Omit --mmproj, keeping --image, to check truthful capabilities and rejection.
```

The independent reference tools are defined by
`src/models/gemma4/tools/reference.CMakeLists.txt`. Copy that file to an ignored
build directory as `CMakeLists.txt`; configure with `GUFO_SOURCE`, `LLAMA_SOURCE`
(the pinned checkout), and `LLAMA_BUILD` (the existing independent build), using
its compiler environment. Build all five executables. Then run
`src/models/gemma4/tools/reference_blas.py --llama-source <pinned-checkout>
--llama-build <existing-build> --output <isolated-directory>
--reference-tool <built-llama-reference.exe>` in the same environment, and copy
the assistant/vision/image executables into that isolated directory. Original
reference sources and DLLs remain unchanged. The script records exact backend
compile/link commands in `build-commands.json`. Use the original DLLs in a
separate directory for performance comparisons.

Numerical reference commands use `GGML_CUDA_DISABLE_FUSION=1`, the installed
ROCm rocBLAS library path, FP32 KV, flash off, eight CPU threads, one sequence,
4K context and the FP32 callback. Example text/window commands:

```powershell
$env:GGML_CUDA_DISABLE_FUSION = '1'
# Set ROCBLAS_TENSILE_LIBPATH to the selected TheRock bin/rocblas/library.
build/gemma4/reference-blas/llama-reference.exe $gemmaTarget "$gemmaFixtures/corpus-tokens.txt" build/gemma4/reference-text.f32 277 - --f32
python src/models/gemma4/tools/compare.py build/gemma4/reference-text.f32 build/gemma4/batch.f32 "$gemmaFixtures/corpus-tokens.txt" --output build/gemma4/text-comparison.json
build/gemma4/reference-blas/llama-reference.exe $gemmaTarget "$gemmaFixtures/corpus-tokens.txt" build/gemma4/reference-window.f32 512 - --f32 --window
```

For sparse-window loss, expand the corpus IDs cyclically to 2052 tokens and pass
that file plus `--positions src/models/gemma4/fixtures/window-positions.txt` to
`compare.py`. For vision, supply RGB8 decoded from `shapes.png` (256x128) to
`llama-vision-reference MMPROJ RGB 256 128 DIRECTORY`. For assistant, use
`llama-assistant-reference TARGET ASSISTANT assistant-tokens.txt DIRECTORY`.
Compare their captured tensors with `compare_layers.py`, limits 0.01/0.1 for
vision and 0.05/0.5 for assistant. For image/window logits, run
`llama-image-reference TARGET corpus-tokens.txt projected.bin
image-suffix-tokens.txt LOGITS 1008` and compare the 17 rows using
`image-comparison-tokens.txt`. Each engine consumes its own matching projector
output; only the weights/artifact set and token history are shared.


The final API matrix passes both sets with and without vision, plus the complete
QAT vision suite with MTP. Replacement answers are `Red` then `Blue`; both ordered
image tests answer `second`. Buffered and streamed Responses agree, including
the conventional newline-only immediate tool continuation, and follow-up/replay
answers correctly read the image in tool history.

Final scalar full-logit files are byte-exact against the pre-batching native
baselines: QAT SHA-256
`f7532b662b264d9360a85302177708e928896e7a6a9cbb85895a0dd3e2a95181`;
conventional
`322517e30ff0aaf427888c8226286dd3b88baadeaba2401de4e28a2b2fce1b77`.
Each file has 277 x 262144 FP32 values (290455552 bytes). The intentional batched
reduction change remains covered by independent full-logit/loss limits.

The CLI image prompt with QAT MTP answers `Red`; raw-mode images are rejected.
The CLI smoke check also corrected an invalid documentation example: `prompt`
does not accept `--context`; serving does. The model guide now uses the actual
prompt interface. Compact final evidence, build/artifact/fixture hashes,
acceptance counts, memory accounting and explicit unmeasured coverage are retained
in [qualification.json](../models/gemma4/qualification.json). Shared fixes are
separate local commits `c9044a8` (reasoning delimiters) and `ddade81` (atomic
image prefill/cache endpoints). No existing-model GPU arithmetic is changed.

## GUI follow-up (2026-10-02)

Added **Gemma 4 31B** to the GUI's existing model-family selector for both
qualified artifact sets. The shared family definition bounds context to
2..4096, fixes one session, disables persistent caching and restricts reasoning
levels to the model default. Selecting the family normalizes unsupported
settings without changing sampling or paths. Explicit family defaults set
Thinking Off, speculative decoding Off and a two-token cap for optional MTP.
Gemma MTP commands include the assistant path and draft cap, without Flash-Next
policy/vocabulary/lookup flags. Vision guidance requires the matching BF16
projector explicitly and describes a blank field as text only.

Validation: all 37 Python GUI tests and all 9 JavaScript DOM tests pass.
The added checks cover both quantization filename forms, sidecar restoration,
family switching, bounded context/session controls, rejection of unsupported
settings, native argument construction and preservation of saved configurations.
File validation and command construction also pass against all six real local
artifacts, with speculative decoding Off and MTP for each target set.
An isolated Flask/Waitress browser smoke check exercised the actual QAT paths,
family defaults, MTP controls and command preview; the server remained stopped
and the user's saved launcher file remained unchanged. This GUI-only follow-up
does not change engine arithmetic or require another model/GPU qualification.

```powershell
.\build\gui-env\Scripts\python.exe -m unittest discover -s tests/tools -p "gui_*_test.py"
node --experimental-vm-modules --test tests/tools/gui_dom_test.mjs
python tools/ci/check-docs.py
```
