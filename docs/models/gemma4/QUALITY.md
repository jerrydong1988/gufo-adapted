# Gemma 4 quality qualification

The two supplied 31B GGUF targets, their matching assistants and BF16 projectors
are qualified separately. Exact identities and provenance are in the model's
artifact manifest. The independent oracle is llama.cpp revision
`1d13fa1c5d6ed9f8cfdc924744b204070c32466d`, TheRock 10.0.0 / Clang 23, FP32 BLAS,
FP32 KV, 4096 context, flash attention off and full SWA cache. Its isolated build
honors FP32 dispatch requests and selects the checkpoint's GELU tanh vision
activation; see `docs/plans/gemma4.md` and `reference_blas.py`.

Limits declared before comparison: full-logit RMSE <= 0.1, maximum absolute error
<= 1, matched next-token perplexity relative error <= 1%; assistant intermediates
RMSE <= 0.05 and maximum error <= 0.5; vision intermediates RMSE <= 0.01 and
maximum error <= 0.1. Finite outputs are required. State restoration, fork,
cancellation recovery and AR/MTP comparisons require exact full-logit equality
when running the same native arithmetic and token history.

## Independent numerical comparisons

Each text row contains all 262144 logits. The 277-token corpus is identical
between candidate and reference within each artifact set. Batched diagnostics
cycle widths 1/8/9/32/33/128. Window checks compare frontiers at positions 20,
1023..1028 and 2049..2052 after the same 2052 input tokens. Image checks insert
72 projected tokens after a 1008-token prefix and compare the image frontier
and 16 subsequent text frontiers. Perplexity is matched next-token loss over
the recorded rows; sparse window/image results are not whole-corpus quality scores.

| Set / route | Full rows | Maximum row RMSE | Maximum error | PPL relative error | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional scalar text | 277 | 0.00592638 | 0.0473521 | 1.55477e-6 | 277/277 |
| QAT scalar text | 277 | 0.000568722 | 0.00265312 | 2.06626e-6 | 277/277 |
| Conventional batched text | 277 | 0.00563335 | 0.0451210 | 1.99374e-6 | 277/277 |
| QAT batched text | 277 | 0.000321345 | 0.00174093 | 1.68095e-6 | 277/277 |
| Conventional window frontiers | 11 | 0.0133574 | 0.0531216 | 2.82743e-6 | 11/11 |
| QAT window frontiers | 11 | 0.00244638 | 0.00936890 | 1.05288e-6 | 11/11 |
| Conventional image/window frontiers | 17 | 0.00114066 | 0.0105993 | 3.51929e-4 | 17/17 |
| QAT image/window frontiers | 17 | 0.000395339 | 0.00225878 | 5.66942e-5 | 17/17 |

Every row is finite and every comparison passes the declared limits. Scalar
corpus PPL is 186.521962 reference / 186.522252 native for conventional and
334.354836 / 334.355527 for QAT. Top-1 is a diagnostic; full logits and loss
determine acceptance. Separate weights are never compared as an equality test.
After batching was added, the unchanged scalar path was rerun for both targets;
all 277 full rows are byte-exact against the recorded native baselines. File
SHA-256 values, precise summaries, executable identities and fixture hashes are
retained in [qualification.json](qualification.json).

| Component | Captured tensors per set | Conventional RMSE / error | QAT RMSE / error |
| --- | ---: | ---: | ---: |
| Assistant, seven proposal steps | 49 | 0.0000852304 / 0.000411987 | 0.0000333239 / 0.000167847 |
| Vision, including all 27 encoder layers | 41 | 0.000679861 / 0.0458984 | 0.000679861 / 0.0458984 |

Assistant seed/proposal IDs match the reference. The encoder outputs match
between supplied projector sets, but their final projected embeddings differ.
RGB8 preprocessing is byte-exact against the independent reference. The CPU
image fixture checks the complete resized RGB SHA-256, dimensions and geometry.

## Native state and verification

Both sets pass exact full-logit comparisons for snapshot, restore, fork and
continuation across two local windows, plus fresh recomputation and cancellation
recovery. Complete KV/hidden/logit digest checks cover image snapshot/fork/
cancellation at the first window and the AR/MTP comparisons below. Drafting and
draft cancellation leave the complete target state unchanged; retries produce
identical proposals.

At prefix position 1023, both sets pass 24 greedy and 24 sampled AR/MTP tokens
with exact full target state, position and sampler RNG. Sampled settings are
temperature 0.8, top-k 64, top-p 0.95, repeat penalty 1.05, frequency penalty
0.1 and seed 42. Tests cover accepted/rejected proposals, all three stop IDs,
natural stop after partial block acceptance, target cancellation and restoration.
The verifier samples the target distribution sequentially; it commits only
actual target selections and the corresponding RNG draws.

## Coverage and limits

Rebuilt Windows CPU checks cover Gemma protocol/image fixtures, shared API,
streaming reasoning filters, atomic image cache boundaries, scheduler and cache
lifecycle, existing Qwen tokenizer/templates and Flash-Next sampling/op contracts.
Actual-GGUF protocol checks cover 66 tokenizer/template cases per set and both
accepted and rejected assistant pairings. Real-model API and common-harness
results are recorded in the [implementation log](../../plans/gemma4.md).

Both common harnesses pass 4/4 with no skips. Both sets pass HTTP text, reasoning
on/off, buffered/streaming equality, usage, tool-definition rejection, truthful
loaded image capabilities, direct images, image replacement/order, Chat history,
image-bearing tool results and stateless Responses replay. QAT also passes the
complete image HTTP suite with MTP enabled. The conventional immediate tool
continuation emits a newline and stops on the recorded prompt; its explicit
follow-up correctly identifies both shapes/colors from the retained image.
The suite preserves that original response and checks its streaming equality,
then requires the semantic follow-up and replay answer.

The supported context is 2..4096 combined text/image tokens with one session.
Disk snapshots, tool definitions/generated tool calls, other Gemma variants and
quants remain disabled. Incoming tool history and image-bearing tool results
are supported. No long-context or concurrency claim follows from these checks.

Linux test targets are wired into bounded CI, but local WSL has no compiler,
CMake or Nix, so Linux execution remains unverified. Windows has no rocprofv3;
GPU timeline/busy-time and ISA profiling remain unmeasured. Performance records
use synchronized wall timings in [BENCHMARKS.md](BENCHMARKS.md).
