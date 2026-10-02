# Shared model operators

Correct, tested CPU reference operators shared by model packages. A new
model starts from these instead of re-deriving every formula; genuinely new
math arrives as a new operator with the same treatment.

## Exactness contract

Each operator documents its **exact formula and accumulation precision** in
`scalar.hpp`. Sharing is only correct when the model's math matches
bit-for-bit. Known divergence examples already in tree:

- RMSNorm: Flash-Next adds epsilon in float32 after narrowing the
  float64 mean; Qwen adds it in float64 and finishes in float64.
- SiLU: Flash-Next multiplies by the reciprocal (`x * (1/(1+exp(-x)))`);
  Qwen divides (`x / (1+exp(-x))`). Different rounding.
- Softmax: Flash-Next's inline MoE form exponentiates in float32; Qwen's
  oracle exponentiates in float64.

Consequences:

- A model adopts an operator only after a parity proof (see below).
- Until then it keeps its private oracle. Private oracles are not tech
  debt; they are the model's numerical truth.
- Every operator header records its adopters. If your model matches,
  adopt it and record the adoption.

## Adoption progression

1. **Correct reference.** The operator exists here with analytic tests in
   `tests/models/common/ops_test.cpp` (hand-computed values, not copies of
   the implementation).
2. **Parity proof.** The adopting model adds a bit-exact differential test
   against its oracle on fixed vectors, e.g.
   `tests/models/qwen38_flash_next/ops_parity_test.cpp`, with recorded
   checksums so later changes are caught.
3. **Measured specialization.** Production GPU kernels stay model-private
   until a second model needs the same operator on the same hardware.
   Sharing a fast path requires the parity test plus end-to-end
   measurements showing no regression for every adopter. Successful
   specializations become the default; no extra switches.

## Backend selection (future)

The intended end state is shape/quantization/hardware dispatch behind
these operators: the planner picks matrix-vector kernels for decode,
matrix-matrix for prefill, and fused or model-specific kernels where
measurement justifies them. That dispatcher does not exist yet; today's
GPU kernels remain model-private by design. Do not build the dispatcher
until two models share a GPU kernel with the parity and measurement
proofs above.

## Adding an operator

- Put the scalar formula in `scalar.hpp`/`scalar.cpp` with its exact
  contract and an initially empty adopter list.
- Add analytic cases to `tests/models/common/ops_test.cpp`.
- If a model adopts it immediately, add that model's parity test and
  record the adoption in the header.
