# Gemma 4 31B

Native package `gemma4` supports the supplied dense 31B instruction-tuned
conventional and QAT GGUF sets on gfx1151. Initial limits are one session and
2..4096 combined text/image tokens. Text, state restoration, assistant/MTP and
vision have separate independent qualification records in [QUALITY.md](QUALITY.md)
and the [implementation journal](../../plans/gemma4.md).

The [artifact manifest](../../../src/models/gemma4/fixtures/artifacts.json)
records all six SHA-256 values, publisher revisions, complete tensor inventories,
storage types and tokenizer/template identities. Weights are kept outside Git.
Qualified target sets:

| Set | Target storage | Assistant | Vision |
| --- | --- | --- | --- |
| Conventional, `experimental/alternate-gemma4-31b` | Q4_K/Q5_K/Q6_K + FP32 | Matching Q8_0 `gemma4-assistant` | Matching BF16 `gemma4v` projector |
| QAT, `experimental/unsloth-gemma4-31b-qat` | Q4_0 + FP32 | Matching Q4_0 `gemma4-assistant` | Matching BF16 `gemma4v` projector |

Filenames alone do not describe storage. Assistant and projector identities are
checked against their set; projectors have different projection weights.
Other Gemma sizes, quants and F16 projectors are outside the qualified contract.

## Run

From the repository root on Windows, build with
`tools/windows/build.ps1 -Target gufo -Jobs 4`. Use the corresponding `gufo`
binary on Linux; Linux execution of this new package is awaiting CI.

```powershell
build/release/gufo.exe prompt --model experimental/unsloth-gemma4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --think off "What is 2+2?"

build/release/gufo.exe serve llm --model experimental/unsloth-gemma4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --context 4096 --think off --speculative off --mmproj experimental/unsloth-gemma4-31b-qat/mmproj-BF16.gguf

build/release/gufo.exe bench --model experimental/unsloth-gemma4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --n-prompt 2048 --n-gen 128 --seed 42
```

For MTP, explicitly add `--speculative mtp --mtp-model <matching-assistant>`.
Draft width is 1..7 with minimum draft count 1. Flash-Next survival/Latin
controllers, other draft backends and prompt lookup are rejected. MTP currently
preserves exact AR tokens, sampler RNG and target state using sequential target
verification; it is slower on the retained workload. See
[BENCHMARKS.md](BENCHMARKS.md) before enabling it for speed.

Vision requires an explicit compatible `--mmproj`. `/v1/models` and `/props`
advertise images only after it loads. Generic chat-mode `prompt --image <file>`
uses the same preparation path as serving. Images in tool results and stateless
Responses replay preserve image order and function-call identity. Tool
definitions and generated tool calls remain disabled; validated incoming tool
history is supported. Raw-mode image prompting is rejected.

Snapshots and fork store actual valid ring rows, hidden state and logits. Disk
snapshots remain disabled. Opening the GUI does not load Gemma; existing GUI
settings and model selections are preserved.

## Reproduce qualification

Configure/build the `gpu-test` preset and `gemma4_validate` target. The tool
supports fixed-token full logits, `--batch`, `--state`, `--assistant`,
`--speculative`, `--multimodal-speculative`, `--vision`, `--image-state`, and
`--harness [ASSISTANT]`. Exact commands and output identities are recorded in
the journal. Model-owned reference tools link only the pinned independent
llama.cpp build; the precision-matched numerical DLL and its original
performance DLL are kept separate.

CPU CI runs `gemma4_protocol_test` (both embedded-template fixture sets) and
`gemma4_image_test` (complete RGB8 hash against independent mtmd preprocessing).
Passing target/assistant paths to the protocol tool also checks actual tokenizer
IDs and sidecar compatibility. `--reject-assistant` checks incompatible pairs.
Small fixed-token corpus, assistant, image-suffix and frontier-position fixtures
are retained alongside the manifest. Python numerical comparisons need NumPy;
template fixture generation needs Jinja2. The real-model HTTP tool needs Pillow
for its second analytic image; use an isolated environment under `build/`.

All production GPU operations remain private to this package. Text prefill uses
bounded FP32 BLAS, scalar decode retains quantized-weight FP32 GEMV, and vision
uses the same private FP32 norm/GELU/BLAS helpers. Future cross-model extraction
and planner candidates are documented in [EXPERIMENTS.md](EXPERIMENTS.md).
