# HTMLNLM Evangelion — Release Notes

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
