# Gemma 4 experiments

| Mechanism | Status | Evidence / decision |
| --- | --- | --- |
| FP32 private scalar quantized GEMV | Baseline | Both targets pass matched logits/PPL. |
| Q8 activation quantization for GEMV | Rejected | Failed the declared FP32 numerical contract; removed. |
| FP32 BLAS for complete image blocks | Retained | Independent encoder and image-language checks pass; preserves bidirectional local masks. |
| Bounded FP32 BLAS text prefill | Retained | Both sets pass 277 full rows and 11 window frontiers; QAT pp2048 gains 4.51064x, pp16 gains 1.70557x, scalar decode control stable. |
| Extract shared GPU operations | Deferred pending proof | Existing Qwen/Flash quantization and reduction contracts differ. No unqualified cross-model kernel adoption. |

`tools/prof/prof.py` was attempted on this Windows installation and reports
`rocprofv3 not on PATH; run inside nix develop`. TheRock supplies no profiler
here. GPU timeline, busy-time and ISA profiling remain unmeasured; wall timings
must not be described as GPU profiles. Context beyond 4K and concurrency beyond
one remain separate work.


## First operation and planning candidates

Gemma target and `gemma4v` already reuse private FP32 RMSNorm, GELU and BLAS
operations, qualified by target full logits/PPL and all encoder intermediates.
Their normalization weights are already converted; reduction order and FP32
boundaries remain explicit. Tied global K/V, proportional RoPE and local
bidirectional image attention remain model-private.

The first cross-model candidate is row-major FP32 matrix multiplication with
transpose/none operands, alpha=1, beta=0 and atomics disabled. Gemma's
`device::Blas` and Qwen3-TTS `hip/speech_decoder_runtime.cpp::F32Gemm` have that
contract. Keep their thin library wrappers private until a common implementation
has independent parity and matched whole-model measurements for both. No TTS
kernel is changed or claimed qualified by Gemma's results. Qwen/Flash integer
activation GEMV and SiLU gates do not match Gemma's FP32 activation/tanh GELU
contract; their shared adoption is explicitly rejected without new proof.

The retained Gemma selection rule is scalar quantized-weight FP32 GEMV for one
row and bounded FP32 BLAS for larger prefill blocks, capped at 128 text rows or
280 image rows. Only image-local layers expose the full image block; text and
all global layers stay causal. Extra ring capacity protects the earliest query
while a block writes KV. An exact snapshot stores physical ring contents and
frontier activations. This supplies concrete shape/mask/precision requirements
for a future planner; no universal graph or additional user switches are added.
