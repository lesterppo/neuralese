# Neuralese — Latent-Space Agent Communication

**Hypothesis: AI agents can communicate through low-dimensional continuous vectors instead of text tokens.**

**Answer: Yes — with training. A 4-layer CNN learns an 8D protocol in 4,000 games (88-100% accuracy). A 70B LLM cannot discover it in one shot. The protocol is architecture-agnostic (works across CNN, LSTM, Attention) and deployable as a 298KB plugin.**

## Project Journey (June 2026)

This project went through three phases:

### Phase 1: Supervised Autoencoders (Paths A, B, C) — FLAWED
The original benchmarks claimed 97.5% exact match and 116,143% MI Gain. These were
inflated by fixed-vocabulary overcapacity and shared-latent confounds. Path A/B used
bottlenecks 17× wider than information content. Path C generated both agents' states
from the same latent factors. All three have been corrected and re-benchmarked.

### Phase 2: Maze RL (v1-v12) — FALSIFIED
Twelve versions of an Observer→Navigator maze experiment. All failed because the
Navigator had an alternative strategy (radar-based wall avoidance). The null-channel
test proved the Observer's communication channel was unused (+8% over noise). Key
insight: communication must be the ONLY path to reward.

### Phase 3: Referential Game — PROVEN
The Sender reads raw function text, encodes it into 8D, and the Receiver identifies
the target from 4 candidates using ONLY the vector. No radar, no recurrence, no
alternative strategy. Result: 88-100% accuracy, null channel at chance (25%).
This is the first genuine proof of emergent Neuralese communication.

## Current Results

| Benchmark | Result | Status |
|-----------|--------|--------|
| Referential game (8D, CNN) | 88-100% accuracy | ✅ PROVEN |
| Minimum viable bottleneck | 3D (96 bits) | ✅ |
| Optimal bottleneck | 8D (256 bits) | ✅ |
| Candidate scaling | 4 max at 8D | ✅ |
| Cross-architecture (CNN/LSTM/Attn) | 6/6 pairs work | ✅ |
| Task routing (16D) | 97.8% accuracy | ✅ DEPLOYABLE |
| Context compression (16D) | 55.6% exact, 2.8× savings | ✅ VIABLE |
| LLM one-shot (Llama 70B) | 20% (chance) | ❌ FAILED |
| LLM one-shot (GPT-OSS 120B) | 40% (marginal) | ⚠️ WEAK |
| Maze RL (v1-v12) | Channel unused | ❌ FALSIFIED |
| Original Path A/B/C claims | Inflated by confounds | ❌ CORRECTED |

## Quick Start

```bash
pip install torch numpy matplotlib

# The proof: referential game (2 minutes)
python3 referential_game.py

# Deployable plugin: train + demo (3 minutes)
python3 hermes_plugin.py train
python3 hermes_plugin.py demo

# Exploration: bottleneck sweep (10 minutes)
python3 exploration.py

# Cross-architecture test (5 minutes)
python3 xarch.py
```

## File Map

### Active (use these)
| File | Purpose |
|------|---------|
| `referential_game.py` | The proof — referential game with CNN Sender → 8D → Receiver |
| `hermes_plugin.py` | Deployable plugin — train, demo, test pipeline |
| `exploration.py` | Systematic sweep: bottleneck dims, candidate counts |
| `xarch.py` | Cross-architecture validation (CNN, LSTM, Attention) |
| `applications_v2.py` | Task routing (97.8%) + context compression (55.6%) |

### Diagnostic (run to understand failures)
| File | Purpose |
|------|---------|
| `maze_audit.py` | Null-channel test + disentanglement on maze |
| `maze_v12_independent.py` | Independent agents, GRU baseline, info plane |
| `path_c_sync.py` | Corrected state sync with overlap sweep |
| `path_b_stress.py` | Open vs closed vocabulary stress test |

### Legacy (historical, known issues)
| File | Original Claim | Issue |
|------|---------------|-------|
| `path_a_bridge_v3.py` | 97.5% exact match | Fixed vocab, 17× overcapacity |
| `path_b_instructions.py` | 98.0% exact match | 6-way edit type is trivial |
| `cross_model_test.py` | "Cross-model communication" | Text roundtrip, not latent |
| `maze_navigator_v5.py` | "Emergent latents std 0.71" | Diversity loss noise, not signal |
| `maze_navigator_v11.py` | Channel noise fix | Navigator ignores Observer |
| `neuralese_bridge.py` | v2 embeddings bridge | Superseded by referential game |
| `demo.py` | Bandwidth proof | Valid but limited scope |

### LLM Integration (experimental)
| File | Purpose |
|------|---------|
| `nv_game.py` | NVIDIA NIM referential game (Llama 70B: 20%) |
| `llm_openrouter.py` | OpenRouter multi-model test (rate limited) |
| `llm_multi_clean.py` | OpenRouter with correct model IDs |
| `llm_referential.py` | Original LLM referential attempt |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ ACTIVE: Referential Game                                 │
│                                                          │
│ Sender (CNN)          Receiver                           │
│ raw text → CNN        hash embeddings + 8D vector        │
│   │                      │                               │
│   ▼                      ▼                               │
│ 8D Neuralese ────────► Categorical(4 candidates)        │
│                                                          │
│ Accuracy: 88-100%   Chance: 25%   Null: 25%             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DEPLOYMENT: Hermes Plugin                                │
│                                                          │
│ LLM text → PyTorch Sender (4KB) → 8D vector             │
│ 8D vector → PyTorch Receiver (4KB) → decoded text → LLM │
│                                                          │
│ The vector NEVER enters the LLM context window.          │
│ Model: 298KB. Savings: 3-8× tokens.                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ FAILED: Maze Observer→Navigator                          │
│                                                          │
│ Observer           Navigator                             │
│ full map → 12D     radar + 12D → movement                │
│                              ↘                           │
│                    Navigator uses radar, ignores z       │
│                    Null channel ≈ Neuralese (+8%)        │
└─────────────────────────────────────────────────────────┘
```

## Key Design Insight

**Communication must be the ONLY path to reward.** The maze failed because the
Navigator could wall-avoid using its local radar. The referential game succeeds
because the Receiver has no other information source — it must use the Sender's
vector or guess randomly.

This is the central lesson for any emergent communication experiment.

## Known Limitations

1. **LLMs cannot one-shot.** The protocol requires training (4,000+ games).
LLM inference alone cannot discover it. Use trained PyTorch adapters.
2. **Format-locked.** The trained encoder expects the exact text format it was
trained on. Changing the prompt format degrades accuracy.
3. **Candidate ceiling.** The 8D protocol maxes at ~4 candidates. More candidates
require larger bottlenecks.
4. **Synthetic data.** All training uses synthetic function descriptions. Real
agent task traces would improve generalization.

## Next Steps

- [ ] Train on real Hermes `delegate_task` logs for production accuracy
- [ ] Embedding-space injection: 8D → LLM residual stream (zero tokens)
- [ ] Multi-agent broadcast: one Sender, N Receivers sharing one vector
- [ ] Dec-POMDP with independently trained agents

## License

MIT
