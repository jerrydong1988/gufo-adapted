# Gemma 4 performance

Initial qualification uses Windows 11 x64, AMD Strix Halo gfx1151, TheRock
ROCm 10.0.0 / Clang 23, Release builds, one session and a 4096-token context.
Artifact identities are in `src/models/gemma4/fixtures/artifacts.json`.

Retained measurements use the same 2048-token prompt and up to 128 generated
tokens, greedy sampling, seed 42, no vision and cold request state. Loading,
warmup, snapshot capture and restore are outside the inference timers. The
assistant is resident for both AR and MTP measurements. Actual completed-token
counts, acceptance and exact output/state parity must accompany timings.

The initial scalar prefill path is approximately 4 tokens/s on the small
integration fixture. This is a debugging observation, not a pp2048/tg128 result.
Production baseline binaries are retained under ignored `build/`.


## QAT release baseline and bounded text batching

Matched 2048-token prompt / 128-token generation, TheRock 10.0.0 Clang 23,
`release` preset (RelWithDebInfo, -O2, NDEBUG), one session, FP32 KV,
4096 context, temperature 0, seed 42. Warmup: 16 prompt tokens and eight forced
continuation tokens, then reset. Timers include synchronization and sampling;
model loading, snapshots and restore are excluded. Each route includes its first
128-row batch-shape initialization cost. Assistant residency is identical.

| Build / route | Prefill tok/s | Decode tok/s | Completed tokens | Drafts / accepted |
| --- | ---: | ---: | ---: | ---: |
| Before text batching, AR | 3.31577 | 3.01759 | 128 | 0 / 0 |
| Bounded 128-row text batching, AR | 14.9562 | 3.00696 | 128 | 0 / 0 |
| Bounded text batching, MTP | Same prefilled frontier | 2.84602 | 128 | 221 / 65 |

Prefill improves 4.51064x; the scalar decode control changes -0.35%. Both builds
produce the same prompt and all 128 greedy output IDs. Within each build AR/MTP
also have exact RNG, positions and full final logits. MTP is 5.35% slower than
AR on this workload; it remains explicit opt-in and has no speedup claim.
The current verifier evaluates committed tokens sequentially. Its 29.41% draft
acceptance does not translate into saved target work.

Raw reports: `build/gemma4/benchmark-{baseline,candidate}-qat.json`.
Their compact summaries and binary/input/output identities are retained in
[qualification.json](qualification.json). This performance A/B covers QAT;
conventional performance is not inferred from these timings.
The model-owned `gemma4_bench` tool regenerates the prompt, writes its complete
IDs and performs the same measurements. A single long baseline/candidate run is
retained; no confidence interval or warmed library-kernel plateau is claimed.


The generic CLI short-prompt control (pp16/tg8, same QAT model, greedy, seed 42,
no assistant, two deterministic fresh runs per build) also improves: prefill
3951.60 ms / 4.04899 tok/s before, 2316.88 ms / 6.90584 tok/s after (1.70557x).
Decode is 3.93206 / 3.94197 tok/s. This control does not replace the retained
pp2048/tg128 result. Loading is excluded; cold load times varied substantially.


The pinned reference's original HIP DLL was measured separately on the same
2048 prompt IDs and 128 native continuation IDs, 4K context, F32 KV, flash off,
128-token prefill chunks, eight CPU threads and the same 16+8 warmup. Its
unmodified optimized quantized activation arithmetic reports 214.678 tok/s
prefill and 5.80126 tok/s continuation evaluation, with native/reference greedy
agreement 126/128. These are fixed-history evaluation timings with argmax work;
the reference is not loading the Gemma assistant. Its precision differs from
Gufo's FP32 activation contract, so this is context for remaining optimization,
not the numerical oracle or a claim of equivalent arithmetic. Raw report:
`build/gemma4/benchmark-reference-standard-qat.json`; both DLL hashes are recorded
in `build/gemma4/build-identities.json`.
