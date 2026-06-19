# AGENTS.md — Neuralese Project Guide for AI Agents

## What This Project Is
Neuralese tests whether AI agents can communicate through low-dimensional
continuous vectors (8-16D) instead of text tokens.

## The Answer (as of June 2026)
**Yes — with training.** A 4-layer CNN learns an 8D protocol in 4,000 games
(88-100% accuracy). LLMs cannot discover it in one shot (20%). The protocol
works across architectures (CNN, LSTM, Attention). It's deployable as a 298KB
PyTorch model that achieves 3-8× token savings.

## Key Files to Read First
1. `README.md` — Full project status, results table, architecture diagrams
2. `referential_game.py` — The proof. Run this first: `python3 referential_game.py`
3. `hermes_plugin.py` — Deployable pipeline: `python3 hermes_plugin.py train && demo`

## How to Run the Proof
```bash
cd ~/neuralese
python3 referential_game.py
```
Expected: 88-100% accuracy, null channel ~25%. Training takes ~2 minutes.

## Project Structure
- **Active files** (production): `referential_game.py`, `hermes_plugin.py`, `exploration.py`, `xarch.py`, `applications_v2.py`
- **Diagnostic files**: `maze_audit.py`, `maze_v12_independent.py`, `path_c_sync.py`, `path_b_stress.py`
- **Legacy files** (known issues): `path_a_bridge_v3.py`, `path_b_instructions.py`, `cross_model_test.py`, `maze_navigator_v*.py`, `neuralese_bridge.py`, `demo.py`
- **LLM test files**: `nv_game.py`, `llm_openrouter.py`, `llm_multi_clean.py`
- **Trained models**: `models/neuralese_plugin.pt` (298KB)

## What Works
- Referential game: 88-100% emergence with 8D CNN
- Minimum bottleneck: 3D (96 bits)
- Cross-architecture: CNN, LSTM, Attention all work
- Task routing: 97.8% from 16D vector
- Context compression: 55.6% exact, 2.8× savings
- Plugin deployment: 298KB model, 3-8× token savings

## What Doesn't Work
- LLM one-shot: Llama 70B gets 20% (chance). GPT-OSS 120B gets 40% (marginal)
- Maze RL (v1-v12): Navigator ignores Observer, uses radar instead
- Original Path A/B/C claims: inflated by fixed vocab + shared-latent confounds

## Key Design Rule
**Communication must be the ONLY path to reward.** If the Receiver has any
alternative information source (radar, recurrence, partial text), it will
use that instead of the channel. The referential game enforces this by
giving the Receiver only hash embeddings that carry no semantic signal.

## Training Requirements
- Batch training (32 games/step) required; single-game REINFORCE fails
- Gradient clipping needed to prevent weight explosion
- 4,000+ epochs for convergence
- Format must match between training and inference exactly
- Higher dims (16D+) need 8,000+ epochs

## APIs Available
- NVIDIA NIM: `NVIDIA_API_KEY` in `~/.hermes/.env`. 60+ models. No rate limit.
  Endpoint: `https://integrate.api.nvidia.com/v1/chat/completions`
- OpenRouter: `OPENROUTER_API_KEY` in `~/.hermes/.env`. Free tier severely rate-limited.
  Endpoint: `https://openrouter.ai/api/v1/chat/completions`

## Git
- Repo: `github.com/lesterppo/neuralese`
- Branch: `main`
- Remote: `origin`
