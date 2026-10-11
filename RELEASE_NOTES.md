# HTMLNLM Evangelion — Release Notes

## Phase 7 — Senses that are learned · 2026-10-11 · Kehai Interim + Kham

**The senses used to be decorative.** Camera, image and microphone fed random, never-trained LittleBit-2 projections. Those were handed to `injectIntoModel(encoded, App)`, and since `App` has no `currentState`, the guard returned silently. Nothing a sense produced ever reached the language model. The mood path half-worked: senses updated the aesthetic vector and T*, which drove the coherence badges and the text/voice streams, but not GENERATE.

**Now senses are prefix tokens.** A sense becomes K=4 vectors in embedding space that the RWKV reads *before* the text, exactly like tokens. Train on pairs, and the caption is literally "what comes after seeing/hearing this":

- **Paired corpus.** Drop `photo.jpg` + `photo.txt`, `clip.wav` + `clip.txt` (matched by name) into the new SENSES panel. Captions join the BPE vocabulary, and a slider sets the share of training steps that use pairs.
- **SenseAdapter.** Features are standardized with running statistics, then projected to K prefixes by a trainable linear map (Adam, 0.1× the model LR). Its gradient comes from the one that already flows back to the embedding: `LanguageModel.forwardVec()` takes a vector instead of a token id, and `backward()` returns its gradient. Exact against finite differences (`node tests/senses.js`).
- **Features.** Deterministic, so they can be tested headless.
  - Images: 8×8 color grid, 8×8 edge map, and a 16-number pre-attentive saliency block (what stands out from the average color, and where).
  - Audio: 24 log-spaced band energies (mean and spread over time), loudness, zero-crossings, brightness, flux, onset rate and duration.
- **Sense gap meter.** Caption loss with a wrong sense minus with the right one. Above 0 means the model is actually looking/listening.
- **GENERATE can start from a sense:** an image or sound file, the live camera frame, or the last ~2 s of microphone. With no prompt it captions what it perceives.
- **Mood path, fixed.** The sensorium's semantic temperature T* now steers sampling temperature in GENERATE (×0.7 to ×1.3, toggle in the sampler). The dead state-nudge was removed.
- **Saved in exports**, so adapters and their statistics survive save/load.

**Measured** (`node tests/senses-learn.js`, synthetic paired corpus, 48D×2L, evaluated on fresh renders the model never saw):
- **Audio:** picks the right caption for 100% of new sounds (chance 25%), and, greedy with no prompt, says *"a high steady whistle."* and *"high beeps, on and off."* word for word.
- **Vision:** names the ball's color correctly for 6 of 8 new images (*"a red ball…", "a blue ball…"*). It **has not learned left/right**: it always says "left". The side word comes ~15 characters after the sense, and carrying that bit that far through the state is the open problem. It picks the right caption out of 8 ~46% of the time (chance 12.5%).

**What it took** (each step measured, not guessed):
1. **Feature standardization.** Without it, the background shared by every image dominates W·f, and after LayerNorm every image looked like the same prefix.
2. **A gentler adapter learning rate.** Adam moves every weight ~LR per step whatever the gradient's size. A wandering prefix teaches the model to ignore it.
3. **The fix below.** Without it, the base model couldn't even memorize the captions.

## Phase 6.2 — the ternary backward, fixed · 2026-10-11 · Kehai Interim

**BitLinear's input gradient was the gradient of a different network.** The forward multiplies by the ternary weights γ·Q(W), but the backward passed back Wᵀg using the float master weights. The two differ by the whole quantization error, so every layer at every timestep sent back a slightly wrong signal. This came in with the Tier 0 fused backward. The input gradient is now γ·Q(W)ᵀg, and the weight gradient stays straight-through (STE).

The old gradient check could not see this: it swapped in a float forward. `TERNARY=1 node tests/gradcheck.js` now keeps the real ternary forward and checks everything upstream (embedding, norms, mixing coefficients) for exact agreement.

**Measured, same model and data** (memorizing 8 repeating sentences, 48D×2L, 300 steps at the default LR 0.02):

| | loss (nats/byte) |
|---|---|
| before the fix | 2.66, stuck at unigram |
| float weights (reference) | 0.88 |
| **after the fix** | **0.33** |

The same fix shipped to HTMLNLM, EvaROSA and Simulacra.

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
