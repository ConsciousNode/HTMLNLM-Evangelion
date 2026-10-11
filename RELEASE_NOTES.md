# HTMLNLM Evangelion — Release Notes

- Speed, Tier 0 (2026-10-11 · Kehai Interim): fused cache-friendly BitLinear backward; Muon rebuilt (Newton–Schulz on the smaller side with cache-friendly loops, same update to 1e-6, ~2x faster; output head on AdamW per the original Muon recipe); **AdamW is now the default**, because on CPU Muon costs O(dim³) per matrix per step. Measured on HTMLNLM, a default training step went from ~104 s to ~11 s on one CPU thread. Gradient check still exact.

## Phase 6.1 — 2026-10-11 · Kehai Interim

**Backward pass fixed and verified.** The same plumbing fixes as HTMLNLM v1.1, plus one Evangelion-specific bug.

- **Residual gradient restored.** The block returned only the `ln1` path and dropped the residual.
- **`ln2.backward()` result now used.** The channel mix's gradient never reached the time mix.
- **Embedding gradient order fixed.** It was taken before `ln_in.backward`, at ~2% of its true size.
- **Token-shift BPTT added.** `prevX`/`prevX2` gradients now reach the previous timestep.
- **PoST clamp mismatch fixed (Evangelion only).** The forward clamps `w_proj + log_w_bias − 0.5`, but the backward tested `w_proj − 0.5`. Channels pinned by the PoST bias were still receiving decay gradient as if they weren't clamped.

Every weight, `mu` and the embedding match finite differences over a 3-step sequence (worst cosine 0.9989). Test: `node tests/gradcheck.js`, zero dependencies.

## Phase 6 — May 2026

**First public release.**

### What shipped

- Complete omnimodal training and inference pipeline in a single HTML file
- RWKVv7Block backbone with BitNet b1.58 ternary quantization and PoST decay gates
- ModRWKVAdapter per-modality bridges (vision, audio, text, spatial)
- ElasticTok visual tokenizer (temporal delta compression)
- SpikeVox audio encoder (Leaky Integrate-and-Fire, event-driven)
- SheafMemory topological contradiction detection via H¹(ℱ) coboundary norm
- BooleanPhaseDynamics semantic thermodynamics with Maxwell's Angel sincerity filter
- AutopoieticOptimizer self-modification loop
- RIFT Endospace holographic state visualization
- MuonOptimizer (quintic Newton-Schulz orthogonalization) for training
- GRPO alignment with configurable reward functions
- JuntoNode swarm orchestration for distributed training
- `.evapip` export format (Evangelion Pip — full checkpoint including adapters, restriction maps, optimizer state)

### Format note

Export format changed from `.pip` → `.evapip` in this release to distinguish Evangelion checkpoints from earlier HTMLNLM formats and to ensure forward compatibility with the EvaROSA runtime (RWKV-8 ROSA integration, in development).

### Architecture

Zero dependencies. No build step. No GPU. No server. Single HTML file.
Runs on any device with a modern browser. Tested on desktop and Android.

### Known limitations

- Pre-trained weights not bundled — model initializes from procedural init
- GRPO reward function is configurable but requires manual tuning per task
- Distributed training via JuntoNode requires a shared network environment

### Credits

Kham (systems architecture) · Kehai Interim (mathematical foundation, BitLinear kernel, SheafMemory topology) · Ed Interim (implementation) · Vael Interim (Phase 6, .evapip format)

---

## Roadmap

**EvaROSA** — RWKV-8 ROSA integration. Neurosymbolic inner monologue channel constrained by SheafMemory topology. A model that cannot gaslight itself. In development.
