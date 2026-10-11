# HTMLNLM Evangelion — Release Notes

## Phase 7.1 — Senses v3: eyes and ears as streams · 2026-10-11 · Kehai Interim + Kham

Kham asked what it would take to tell cat from dog, or to hear speech, and whether the senses could get the *capability*, with understanding left to training. This is that step, measured honestly, negative results included.

**Ears: 16 kHz, 40 mel bands, streamed.** 25 ms frames every 10 ms, 4 frames per token: one token per 40 ms, up to 4 s. The order of sounds is kept, so "dog" and "god" are different streams. The old ears averaged a whole clip into one vector.

**Eyes: 32×32, 16 patch tokens, learned filters.** Each 8×8 cell becomes a token with:
- downsampled colour, plus responses to 32 texture filters learned from your own unlabeled images (ZCA whitening + k-means, Coates & Ng 2011);
- a learned position.

The filters are learned at COMPILE BPE from the images loaded so far, and saved with the model.

**Context with coarse structure.** The vector that keeps a sense in view while Eva writes now carries:
- per-quadrant colour and texture for images;
- per-quarter spectra for sound.

It is scaled by 1/√(summary size). Without that, Adam grew it about 400× faster than a token embedding and it drowned the language: |context| swung between 7 and 14 against an embedding of 14, and captions decayed into *"a blue creenreenreen"*.

**Capability, measured with a linear probe** (`node tests/senses-probe.js`: a plain classifier on the raw features of fresh renders, no language model involved):

| | old senses (v2) | **v3** |
|---|---|---|
| vowel ORDER, 6-way ("a then i" vs "i then a"; an order-blind ear tops out at 50%) | 83% | **100%** |
| large shapes, 4-way | 78% | 74% |
| small shapes (~1/8 of frame), 4-way | 40% | 39–41% |

- **Ears: a clear capability win.** Order is what speech is made of.
- **Eyes: an honest negative result.** At this scale, resolution alone did not buy detail. One layer of 4×4 filters read linearly can't assemble "a cross" from fragments a few pixels across: it memorizes (100% on training data) but doesn't generalize. Recognising objects needs depth: learned features of features, trained end-to-end through the exact gradient path that already exists. That's the next experiment.

**Two benchmark bugs found and fixed while measuring:**
1. In fixed-layout vowel clips, frame cut-offs at the end gave away which vowel came second.
2. The class and that example's "random" pitch/size were drawn back-to-back from the same simple generator, so the pitch encoded the class. Classes now come from an independent generator.

Results reported before these fixes (including Phase 7's) may be somewhat flattered; the tables here are after them.

**Cost:** an image is now 16 tokens and a second of sound 24, so training is slower per example. Learning also starts slower in the app: the sense gap stays near 0 for the first few minutes, versus ~+0.2 with v2. Captioning end-to-end is slow and noisy at 48D with one example per step (see `tests/senses-learn.js`, `TASK=shapes|vowels|balls`, `SMALL=1`). Expect it to need more examples and steps than v2 did for coarse captions.

**Compatibility:** v2 sense adapters in older exports are detected, and the senses restart fresh (the language model loads as before).

## Phase 7 — Senses that are learned · 2026-10-11 · Kehai Interim + Kham

**The senses used to be decorative.** Camera, image and microphone fed random, never-trained LittleBit-2 projections. Those were handed to `injectIntoModel(encoded, App)`, and since `App` has no `currentState`, the guard returned silently. Nothing a sense produced ever reached the language model. The mood path half-worked: senses updated the aesthetic vector and T*, which drove the coherence badges and the text/voice streams, but not GENERATE.

**Now senses are prefix tokens plus a context.** A sense becomes K=4 vectors in embedding space that the RWKV reads *before* the text, exactly like tokens. It also contributes a **context vector** that is added to every token's input while the sense is present, so what the model perceives stays in view instead of having to be remembered. Train on pairs, and the caption is literally "what comes after seeing/hearing this":

- **Paired corpus.** Drop `photo.jpg` + `photo.txt`, `clip.wav` + `clip.txt` (matched by name) into the new SENSES panel. Captions join the BPE vocabulary, and a slider sets the share of training steps that use pairs.
- **SenseAdapter.** Features are standardized with running statistics, then projected to K prefixes by a trainable linear map (Adam, 0.1× the model LR). Its gradient comes from the one that already flows back to the embedding: `LanguageModel.forwardVec()` takes a vector instead of a token id, and `backward()` returns its gradient. Exact against finite differences (`node tests/senses.js`).
- **Features.** Deterministic, so they can be tested headless.
  - Images: 8×8 color grid, 8×8 edge map, and a 16-number pre-attentive saliency block (what stands out from the average color, and where).
  - Audio: 24 log-spaced band energies (mean and spread over time), loudness, zero-crossings, brightness, flux, onset rate and duration.
- **Sense gap meter.** Caption loss with a wrong sense minus with the right one. Above 0 means the model is actually looking/listening.
- **GENERATE can start from a sense:** an image or sound file, the live camera frame, or the last ~2 s of microphone. With no prompt it captions what it perceives. With the camera or mic live, the context is refreshed every 8 tokens, so the scene keeps steering the words as they're written.
- **Mood path, fixed.** The sensorium's semantic temperature T* now steers sampling temperature in GENERATE (×0.7 to ×1.3, toggle in the sampler). The dead state-nudge was removed.
- **Saved in exports**, so adapters and their statistics survive save/load.

**Measured** (`node tests/senses-learn.js`, synthetic paired corpus, 48D×2L, evaluated on fresh renders the model never saw):
| | prefix only | **prefix + context** |
|---|---|---|
| vision, exact captions of new images (greedy, no prompt) | 2 of 8, always "left" | **6 of 8, left/right 8 of 8** |
| vision, picks the right caption of 8 (chance 12.5%) | ~46% | **75–88%** |
| audio, exact captions of new sounds | 2 of 4 | **3 of 4** |
| audio, right caption by step 200 (chance 25%) | 42% | **67%** |

With context it says *"a red ball on the right."*, *"a blue ball on the left."*, *"low beeps, on and off."*, *"a high steady whistle."* word for word about things it never saw or heard. Its misses: green read as yellow twice (they share a strong green channel in these renders), and a low hum called "a low steady whistle".

**Why the context was needed (diagnosed, not guessed).** With the prefix alone, vision learned color but never left/right. Swapping the caption order ("left: a red ball.") flipped it: left/right perfect, color lost. Whichever detail sat furthest from the sense faded, because the model *sees* both and the memory is what runs out. The context vector keeps the sense present for the whole caption.

**What it took** (each step measured, not guessed):
0. **A context vector**, zero at init so it adds no noise until it proves useful (see above).
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
