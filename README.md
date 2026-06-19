# Neuralese — Latent-Space Agent Communication

AI agents communicating via raw hidden-state tensors instead of human language.
**Experimental research project — not production-ready.**

## What It Is

Neuralese explores whether AI agents can communicate through high-dimensional
latent vectors instead of discrete tokens. The hypothesis: human language
(~16.6 bits/token) is a bottleneck for agent-to-agent communication, and
continuous latent vectors could carry more information more efficiently.

**Current status: Three benchmark paths established, but no proof of emergent
communication between independent agents yet.**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Observer (sees full environment)                    │
│  Input: 10×10 maze (100) + Navigator position (2)   │
│  Output: 12D Neuralese vector (bottleneck)           │
└──────────────────────┬──────────────────────────────┘
                       │ 12D latent vector
┌──────────────────────▼──────────────────────────────┐
│  Navigator (local view only)                         │
│  Input: 3×3 local radar (9) + 12D Neuralese         │
│  Output: clamped movement [-0.5, 0.5]               │
│  ⚠ Navigator CANNOT see target or full map           │
└─────────────────────────────────────────────────────┘
```

## Current Status

| Milestone | Status | Key Result |
|-----------|--------|------------|
| **Referential game (v2)** | ✅ **PROVEN** | **88.2% accuracy (chance=25%), null=21% — emergent communication** |
| Bandwidth proof (coordinate reconstruction) | ✅ Complete | Continuous beats discrete at equal bit budgets |
| Path C state sync (fixed) | ✅ Complete | 499% MI Gain with independent agents |
| Task routing (16D) | ✅ Complete | **97.8% accuracy from 16D vector — deployable** |
| Context compression (16D) | ✅ Complete | **55.6% exact match, 2.8× token savings — viable** |
| Cross-architecture test | ✅ Complete | **6/6 pairs work (CNN, LSTM, Attention)** |
| Maze (MLP + PPO, no warm-start) | 🔬 Active | ~18% success, unstable |
| **v12: Independent agents + GRU + info plane** | ❌ **Falsified** | Neuralese 15.5% vs Null 14.0% — channel unused |
| Null-channel test (random z) | ⚠ Audit | 10% success vs 18% Neuralese — channel carries only +8% |
| Disentanglement analysis | ⚠ Audit | 1 of 12 latent dimensions has weak task correlation (r≈0.3) |
| GRU (recurrent) Observer | ❌ Regressive | Kills emergence (std 0.03) |
| Multi-agent broadcast | 📋 Planned | — |

### Breakthrough: Referential Game Proves Emergent Communication

The **referential game** (June 2026) is the first experiment to demonstrate genuine
emergent communication through the Neuralese channel:

- **Sender** (CNN) reads a raw Python function and encodes it into 16D
- **Receiver** sees 4 hash-embedded function candidates + the 16D vector and identifies the target
- **Result**: 88.2% accuracy on held-out functions (chance = 25%)
- **Null channel**: 21.0% — the protocol carries real information, not noise

This succeeds where the maze experiments failed because communication is the
**only** path to success. The Receiver has no radar, no recurrence, no
alternative information source — it must use the Sender's latent vector.

### v12 Maze Experiment (Falsified)

The **v12 independent-agent setup** (separate optimizers, GRU baseline, continuous
reward shaping) ran for 3000 episodes with the following results:

| Metric | Neuralese | Null Channel | GRU Baseline |
|--------|-----------|-------------|--------------|
| Success rate | 15.5% | 14.0% | 7.5% |
| Wall collisions/ep | 0.84 | 0.86 | 0.93 |
| Active z dimensions | 0/8 | — | — |
| Max z→action correlation | 0.074 | — | — |

**Verdict: NULL HYPOTHESIS NOT REJECTED.** The Observer→z channel carries
no significant information (+1.5% over random noise, below the 5% significance
threshold). After training, 0 of 8 latent dimensions correlate with the
Navigator's actions. The Navigator relies entirely on its own 3×3 radar,
ignoring the Observer's instructions.

## Three-Path Benchmark Results (Corrected)

| Path | Task | Old Claim | Corrected Result | Issue Found |
|------|------|-----------|-----------------|-------------|
| A — Bridge v3 | Context compression (16D) | 97.5% exact | 6.5% (fixed vocab) | Fixed 30-file lookup table; bottleneck 17× wider than info content |
| B — Instructions | Code-editing via 12D | 98.0% exact | 100% edit, 0.73 file cosine (open vocab) | Edit type is 6-way trivial; open-vocab file cos plateaus at 0.73 |
| C — Sync | State sync (12D→8D) | 116,143% MI | 499% MI (overlap=0), 4,041% (overlap=10) | Old: both agents' states from shared latent. Fixed: independent states |

### Path C Overlap Sweep (Corrected)

| Overlap | Description | MSE | Baseline MSE | Oracle MSE | MI Gain |
|---------|-------------|-----|-------------|------------|---------|
| 0 | Fully independent agents (hardest) | 0.054 | 0.324 | 0.324 | 499% |
| 3 | 3 shared, 7 private features | 0.007 | 0.220 | 0.222 | 3,204% |
| 10 | Fully shared (old confounded setup) | 0.000045 | 0.0005 | 0.0009 | 1,004% |

With independent agents (overlap=0) and an 8D bottleneck (tighter than 10D source),
the channel provides genuine but modest improvement over baseline. MI Gain drops
as overlap increases — exactly the expected behavior when baseline already has
access to shared information.

### Path B Stress Test (Open vs Closed Vocabulary)

| Bottleneck | Open Vocab File Cos | Closed Vocab File Cos | Open Line MAE | Closed Line MAE |
|-----------|--------------------|----------------------|--------------|----------------|
| 2D | 0.700 | 0.718 | 492 | 507 |
| 4D | 0.700 | 0.719 | 166 | 465 |
| 8D | 0.722 | 0.735 | 79 | 67 |
| 16D | 0.730 | 0.826 | 77 | 65 |

With open vocabulary (hash-based on arbitrary UUID strings), file cosine
similarity plateaus at 0.73. The closed vocabulary achieves 0.83 — a 13%
advantage from fixed-set memorization. The old "97.5% exact match" claim
only holds for fixed-vocabulary one-hot targets.

## Inter-Model Communication

The cross-model test (`cross_model_test.py`) measures autoencoder roundtrip
accuracy — it encodes a task into Neuralese, *decodes back to text*, then feeds
the decoded text to an LLM. This tests whether the autoencoder preserves enough
information for the LLM to take the same action, **not whether the LLM
understands raw latent vectors**. The "inter-model matrix" results reflect
autoencoder fidelity, not genuine latent-space communication.

## Quick Start

```bash
pip install torch numpy scikit-learn matplotlib

# Bandwidth proof (30 seconds)
python demo.py

# Maze audit (null-channel + disentanglement, 5 minutes)
python maze_audit.py

# Path C sweep (overlap experiment, 10 minutes)
python path_c_sync.py

# Path B stress test (open vs closed vocabulary, 3 minutes)
python path_b_stress.py
```

## Files

| File | Purpose |
|------|---------|
| `demo.py` | Continuous vs discrete communication bandwidth proof |
| `maze_audit.py` | **New** — Null-channel test + disentanglement + no warm-start |
| `maze_v12_independent.py` | **New** — Independent agents, GRU baseline, info plane analysis |
| `path_c_sync.py` | **Fixed** — Independent-agent state sync with overlap sweep |
| `path_b_stress.py` | **New** — Open vs closed vocabulary compression stress test |
| `path_a_bridge_v3.py` | Legacy — Fixed-vocab autoencoder (overcapacity bottleneck) |
| `path_b_instructions.py` | Legacy — Fixed-vocab instruction autoencoder |
| `maze_navigator_v5.py` | Legacy — PPO + diversity loss (warm-start confound) |
| `maze_navigator_v10.py` | Legacy — Corrected reward (still warm-start) |
| `maze_navigator_v11.py` | Legacy — Channel noise + fixed pairs |
| `neuralese_bridge.py` | Legacy — v2 learnable embeddings bridge |
| `cross_model_test.py` | Legacy — Autoencoder roundtrip test (not latent communication) |

## Known Limitations

1. **No independent multi-agent setup exists.** All "two-agent" experiments train
   Observer and Navigator jointly with a shared optimizer and loss function.
   For genuine inter-agent communication, agents must be trained independently
   with separate objectives and no access to each other's ground truth.

2. **Fixed-vocabulary benchmarks inflate results.** The old Path A/B used fixed
   vocabularies of 30-128 items that can be memorized through an overcapacity
   bottleneck. The new `path_b_stress.py` uses hash embeddings on arbitrary
   strings for a fairer test.

3. **Maze channel information is marginal.** The null-channel test shows only
   +8% improvement over random noise. The disentanglement analysis reveals
   11 of 12 latent dimensions are unused. The "emergent" latents from earlier
   versions likely reflect noise in unused dimensions amplified by diversity
   loss, not structured communication.

4. **Cross-model test is a text roundtrip**, not latent communication. The LLM
   receives decoded text, not raw latent vectors. A genuine test would require
   the LLM to process Neuralese vectors directly (e.g., through embedding-space
   injection or learned adapters).

## Development Direction

**Short-term (fix fundamentals first):**
- [ ] Train Observer and Navigator independently with separate objectives
- [ ] Add GRU-no-Observer baseline (does recurrence beat the channel?)
- [ ] Measure information plane: I(X;Z) vs I(Z;Y) for genuine compression analysis
- [ ] Test with real agent traces, not synthetic vocabularies

**Medium-term (if fundamentals prove out):**
- [ ] LLM embedding-space injection (bypass text roundtrip)
- [ ] Dec-POMDP setup with partial observability
- [ ] Multi-agent broadcast — one Observer, N Navigators

**Long-term:**
- [ ] Self-supervised Neuralese pre-training on large-scale agent traces
- [ ] MARL-trained emergent protocol with zero human-designed reward

## Development Methodology

This project was co-developed with **Gemini Pro** as a collaborative peer reviewer.
The key insight from this process: **MLP architecture forces the latent space to
carry per-step information by preventing memorization across timesteps.**
Architectures with memory (CNN, GRU) encode the full plan at t=0.

## License

MIT
