# Qwen3.8 27B on Windows

Windows evidence belongs here. The upstream [benchmarks](BENCHMARKS.md),
[quality](QUALITY.md) and [experiments](EXPERIMENTS.md) describe Linux and remain
identical to official Gufo.

## Pinned prompt capture deferred (October 23, 2026)

Official #688 is in ancestry, but its pinned-capture optimization is explicitly
reverted after nine matched long-context timing cells fail on Windows/TheRock
10.0.0. The independent DFlash2 upload-failure draining fix and its unchanged
upstream lifetime test remain included. See the
[sync/deferral record](../../WINDOWS-UPSTREAM-SYNC.md#pr-688-windows-dflash2-continuation-latency)
for exact identities, per-phase ranges and reintroduction requirements.

The rejected full import passes native Q4_K_XL numerical prefill/captured-row
and exact AR/DFlash2 cancel/resume checks. Both HTTP modes individually pass
long-context, text/thinking/tool/image RAM replay and disk restart/spacing.
Actual DFlash2 drafts are recorded. These are correctness observations, not
a performance or complete cross-build continuation pass.

Two candidate runs and three baseline runs retain equal effective 4 GiB RAM,
staging/disk capacities, 32K context, two sessions, identical model paths and
all 143 runtime files. The fully cached sampled replay has identical output,
12,454 prompt tokens, seven completion tokens and 11/five proposed/accepted
drafts. Candidate wall time is 405.42-419.80 ms versus baseline 351.75-358.76 ms;
restore is 20.77-21.44 ms versus 15.27-16.61 ms. The 5%/3 ms gates are unchanged.

An image continuation compares unequal cancellation frontiers (273/278 cached
tokens) and produces different seeded reasoning. Matched-frontier image probes
produce exact output/work on both builds and reproduce both output variants on
baseline. The original full comparison stays failed. Native lifetime testing
confirms that failed uploads drain before caller data is reused, for both
pageable and pinned sources, and that context injection recovers.

Evidence is retained under `build/sync-688-2026-10-23/` in the task checkout.
Cold-prompt/benchmark and broader optimization qualification stopped after the
deferral decision. No speedup or full Windows performance pass is claimed.
