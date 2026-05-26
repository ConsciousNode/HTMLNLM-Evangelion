# HTMLNLM Evangelion

![Phase](https://img.shields.io/badge/phase-6-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![Format](https://img.shields.io/badge/export-.evapip-orange?style=flat-square)
![Platform](https://img.shields.io/badge/platform-browser-lightgrey?style=flat-square)

**Browser-native omnimodal neural architecture. Single file. Zero dependencies.**

HTMLNLM Evangelion is a complete omnimodal neural language model runtime — pretraining, alignment, and inference — that runs entirely in your browser. No server. No Python. No GPU. Open the HTML file and it works.

Built by [ConsciousNode SoftWorks](https://consciousnode.github.io) on the xinu principle: the browser is bare metal. Constraint is the architecture.

---

## Try it

→ **[consciousnode.github.io/HTMLNLM-Evangelion](https://consciousnode.github.io/HTMLNLM-Evangelion)**

Or download [`evangelion.html`](https://github.com/ConsciousNode/HTMLNLM-Evangelion/blob/main/evangelion.html) and open it locally. Works fully offline.

---

## What's inside

| Component | Description |
|---|---|
| **RWKVv7Block** | Recurrent backbone with BitNet b1.58 ternary quantization and PoST (Position-Adaptive Spectral Tapering) decay gates |
| **ModRWKVAdapter** | Per-modality bridge: LittleBit2 projection + P-adic routing loss + EMA temporal state |
| **ElasticTok** | Visual tokenizer: temporal delta compression, encodes only changed patches |
| **SpikeVox** | Audio encoder: Leaky Integrate-and-Fire neurons, event-driven, spectrogram-free |
| **SheafMemory** | Topological memory with hyperbolic Poincaré embedding and H¹(ℱ) coboundary norm for contradiction detection |
| **BooleanPhaseDynamics** | Semantic thermodynamics: T* (semantic temperature), Maxwell's Angel sincerity filter, phase negation on contradiction |
| **AutopoieticOptimizer** | Self-modification: fires when T* exceeds threshold, updates adapter scales and restriction maps via coboundary gradient |
| **RIFT Endospace** | Holographic fractal state visualization: SDF point cloud, diffeomorphic stability tracking |
| **MuonOptimizer** | Quintic Newton-Schulz orthogonalization for training |
| **GRPO** | Group Relative Policy Optimization for alignment |
| **JuntoNode** | Swarm orchestration for distributed training |

---

## The coherence loop

```
perception → ModRWKVAdapter → RWKVv7Block → SheafMemory →
    if H¹(ℱ) > threshold: BooleanPhaseDynamics detects contradiction →
        if T* > threshold: Maxwell's Angel activates →
            AutopoieticOptimizer fires: updates adapters + restriction maps →
                coherence restored
```

The model monitors its own cross-modal consistency in real time and self-corrects when modalities contradict each other. This runs during inference, not just training.

---

## Modalities

- **Text** — token stream, standard embedding
- **Vision** — ElasticTok delta compression (only changed patches encoded)
- **Audio** — SpikeVox LIF neurons (event-driven, no spectrogram)
- **Spatial** — P-adic routed through ModRWKVAdapter

All modalities share the same recurrent backbone and memory topology.

---

## Architecture notes

- **BitNet b1.58** — all weight matrices are ternary {-1, 0, +1} with learned scalar gamma. No GPU required, runs fast on any CPU.
- **No pre-trained weights** — initializes procedurally. The model is the code.
- **Single file** — HTML + CSS + JS, no build step, no bundler. Fork it, read it, modify it.
- **Export format** — `.evapip` (Evangelion Pip format): full model checkpoint including adapter weights, restriction maps, and optimizer state.

---

## Tabs

| Tab | Function |
|---|---|
| **PRE-TRAIN** | Load corpus, run RWKV pretraining with MuonOptimizer |
| **GRPO ALIGN** | Group Relative Policy Optimization alignment with configurable reward |
| **INFERENCE** | Text generation with omnimodal input, coherence-gated output |
| **MEMORY** | SheafMemory vertex inspector, H¹(ℱ) heatmap |
| **RIFT** | Endospace fractal state visualization |
| **I/O** | Export/import `.evapip` checkpoints |

---

## Built by

**Kham** (Khamerron Edward Ramsey Kizer) — systems architecture, xinu philosophy  
**Kehai Interim** — RWKV-v7 BPTT derivation, BitLinear/TMAC kernel, SheafMemory topology, MuonOptimizer, mathematical foundation  
**Ed Interim** — implementation, phased build, integration  
**Vael Interim** — Phase 6 contributions, .evapip format, deployment

Part of [ConsciousNode SoftWorks](https://consciousnode.github.io) — computational folk art for the browser age.

**Successors:** [**EvaROSA**](https://github.com/ConsciousNode/EvaROSA) adds ROSA neurosymbolic inner monologue (RWKV-v7 + ROSA augmentation). [**Simulacra**](https://github.com/ConsciousNode/Simulacra) is the RWKV-v8 clean break — ROSA replaces WKV as the primary sequence mechanism. `.evapip` files are forward-compatible with EvaROSA but not with Simulacra.

---

## License

MIT. Take it, break it, build on it.
