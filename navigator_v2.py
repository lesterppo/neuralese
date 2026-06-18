"""
Neuralese Blind Navigator v2 — Supervised Warm-Start + RL Fine-Tuning
Phase 1: Pre-train spatial encoding (Observer encodes state → Navigator reconstructs target)
Phase 2: RL fine-tune for navigation (Observer encodes → Navigator outputs movement)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIG ---
HIDDEN = 128
LATENT_DIM = 8
VOCAB = 32
TOKENS = 20
MAX_STEPS = 20
WARMUP_STEPS = 2000
RL_EPISODES = 2000
BATCH = 32
LR = 1e-3
RL_LR = 3e-4
ENTROPY_COEF = 0.005
TARGET_THRESH = 0.05

# --- MODELS ---

class Observer(nn.Module):
    """Sees [target_x, target_y, pos_x, pos_y] → outputs 8D Neuralese vector"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),  # Stable distribution
        )

    def forward(self, state):
        return self.net(state)

class Navigator(nn.Module):
    """Receives 8D Neuralese → outputs movement (dx, dy)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, latent):
        return self.net(latent)

class Reconstructor(nn.Module):
    """Receives 8D Neuralese → outputs reconstructed target position (warm-start only)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, latent):
        return self.net(latent)

# Text baseline versions (simpler — no warm-start needed, just compare)
class TextObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = TOKENS
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, TOKENS * VOCAB),
        )

    def forward(self, state):
        return self.net(state).view(-1, TOKENS, VOCAB)

class TextNavigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.net = nn.Sequential(
            nn.Linear(TOKENS * HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, tokens_or_soft):
        if tokens_or_soft.dim() == 2:
            emb = self.embed(tokens_or_soft).view(-1, TOKENS * HIDDEN)
        else:
            emb = torch.matmul(tokens_or_soft, self.embed.weight).view(-1, TOKENS * HIDDEN)
        return self.net(emb)

# --- PHASE 1: WARM-START ---

def warm_start_spatial(observer, navigator, steps=WARMUP_STEPS):
    """Pre-train: Observer encodes state → Navigator outputs movement toward target"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=LR)
    history = []

    for step in range(steps):
        targets = torch.rand(BATCH, 2) * 2 - 1
        positions = torch.rand(BATCH, 2) * 2 - 1  # Random positions
        state = torch.cat([targets, positions], dim=1)
        target_movement = targets - positions  # What Navigator should output

        latent = observer(state)
        movement = navigator(latent)  # [B, 2] — should match target_movement

        loss = nn.MSELoss()(movement, target_movement) + 0.001 * torch.norm(latent, p=2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())

        if step % 500 == 0:
            with torch.no_grad():
                # Test: if Navigator gets correct latent, can it reach target in 1 step?
                test_pos = positions + movement
                test_dist = torch.norm(test_pos - targets, dim=1).mean().item()
            print(f"  Warm-up step {step:5d}: loss={loss.item():.6f}  1-step_err={test_dist:.4f}")

    return history

# --- PHASE 2: RL FINE-TUNING ---

def rl_fine_tune(observer, navigator, episodes=RL_EPISODES):
    """REINFORCE: Observer communicates → Navigator outputs movement → reward = -final_distance"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    history = {"reward": [], "steps": [], "success": []}

    for ep in range(episodes):
        targets = torch.rand(BATCH, 2) * 2 - 1
        positions = torch.zeros(BATCH, 2)
        active = torch.ones(BATCH, dtype=torch.bool)
        ep_log_probs = []
        ep_rewards = []

        for step in range(MAX_STEPS):
            if not active.any():
                break

            state = torch.cat([targets, positions], dim=1)
            latent = observer(state)

            # Navigator outputs movement with exploration noise
            action_mean = navigator(latent)
            action_std = torch.ones_like(action_mean) * 0.3
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            new_positions = positions + action
            new_positions = torch.clamp(new_positions, -1, 1)

            dists = torch.norm(new_positions - targets, dim=1)
            rewards = -dists

            just_finished = (dists < TARGET_THRESH) & active
            active = active & ~just_finished

            ep_log_probs.append(log_prob)
            ep_rewards.append(rewards)
            positions = new_positions

        n_steps = len(ep_rewards)

        # Compute returns (discounted)
        returns = []
        G = torch.zeros(BATCH)
        gamma = 0.95
        for t in reversed(range(n_steps)):
            G = ep_rewards[t] + gamma * G
            returns.insert(0, G.clone())

        # REINFORCE loss
        policy_loss = 0.0
        for t in range(n_steps):
            advantage = returns[t] - returns[t].mean()
            policy_loss = policy_loss - (ep_log_probs[t] * advantage.detach()).mean()

        policy_loss = policy_loss / n_steps

        opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step()

        total_reward = sum(r.sum().item() for r in ep_rewards)
        history["reward"].append(total_reward / BATCH)
        history["steps"].append(n_steps)
        successes = ((positions - targets).norm(dim=1) < TARGET_THRESH).sum().item()
        history["success"].append(successes / BATCH)

        if ep % 400 == 0:
            print(f"  RL ep {ep:4d}: avg_reward={total_reward/BATCH:.4f} "
                  f"steps={n_steps} success_rate={successes/BATCH:.2f}")

    return history

# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=100):
    """Evaluate without exploration noise"""
    total_steps = 0
    total_final_dist = 0.0
    successes = 0

    for _ in range(num_eps):
        target = torch.rand(2) * 2 - 1
        pos = torch.zeros(2)

        for step in range(MAX_STEPS):
            state = torch.cat([target, pos]).unsqueeze(0)
            with torch.no_grad():
                latent = observer(state)
                action = navigator(latent).squeeze(0)

            pos = torch.clamp(pos + action, -1, 1)
            dist = torch.norm(pos - target).item()

            if dist < TARGET_THRESH:
                successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += MAX_STEPS

        total_final_dist += dist

    return {
        "success_rate": successes / num_eps,
        "avg_steps": total_steps / num_eps,
        "avg_final_dist": total_final_dist / num_eps,
    }

# --- VISUALIZATION ---

def plot_all(warmup_hist, rl_hist, neur_eval, out_dir):
    """3-panel summary: warm-up loss, RL reward, eval bars"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(warmup_hist, alpha=0.5, color='blue')
    ax.set_title("Phase 1: Spatial Warm-Start")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rl_hist["reward"], alpha=0.7, color='green', label="Avg Reward")
    ax2 = ax.twinx()
    ax2.plot(rl_hist["success"], alpha=0.5, color='orange', label="Success Rate")
    ax.set_title("Phase 2: RL Navigation")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax2.set_ylabel("Success Rate")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    metrics = ["Success\nRate", "Avg\nSteps", "Final\nDist"]
    vals = [neur_eval["success_rate"], neur_eval["avg_steps"], neur_eval["avg_final_dist"]]
    colors = ['green', 'blue', 'red']
    bars = ax.bar(metrics, vals, color=colors, alpha=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}" if val < 1 else f"{val:.1f}", ha='center', fontweight='bold')
    ax.set_title("Evaluation (100 episodes)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Neuralese Blind Navigator — Warm-Start + RL", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "navigator_v2_results.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

def plot_trajectories(observer, navigator, out_dir, num=5):
    """Plot Agent B's path to target"""
    fig, axes = plt.subplots(1, num, figsize=(num*4, 4))

    for i in range(num):
        ax = axes[i]
        target = torch.rand(2) * 2 - 1
        pos = torch.zeros(2)
        path = [pos.clone().numpy()]

        for step in range(MAX_STEPS):
            state = torch.cat([target, pos]).unsqueeze(0)
            with torch.no_grad():
                latent = observer(state)
                action = navigator(latent).squeeze(0)
            pos = torch.clamp(pos + action, -1, 1)
            path.append(pos.clone().numpy())
            if torch.norm(pos - target).item() < TARGET_THRESH:
                break

        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(path[0, 0], path[0, 1], c='green', s=80, marker='o', label='Start', zorder=5)
        ax.scatter(target[0].item(), target[1].item(), c='red', s=120, marker='*', label='Target', zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=60, marker='s', label='End', zorder=5)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title(f"T=({target[0]:.2f},{target[1]:.2f}), {len(path)-1} steps")
        if i == 0: ax.legend(fontsize=7)

    plt.suptitle("Neuralese Navigator — Agent B Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "navigator_v2_trajectories.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NEURALESE BLIND NAVIGATOR v2 — Warm-Start + RL")
    print("=" * 60)

    # Phase 1: Warm-start (Observer + Navigator trained to output correct movement)
    print("\n[Phase 1] Spatial Language Pre-training...")
    observer = Observer()
    navigator = Navigator()
    warmup_hist = warm_start_spatial(observer, navigator)

    # Phase 2: RL fine-tune
    print("\n[Phase 2] RL Navigation Fine-tuning...")
    rl_hist = rl_fine_tune(observer, navigator)

    # Evaluate
    print("\n" + "=" * 40)
    print("EVALUATION (100 episodes):")
    neur_eval = evaluate(observer, navigator)
    print(f"  Success Rate:  {neur_eval['success_rate']:.1%}")
    print(f"  Avg Steps:     {neur_eval['avg_steps']:.2f}")
    print(f"  Avg Final Dist:{neur_eval['avg_final_dist']:.4f}")

    # Visualize
    print("\nGenerating plots...")
    plot_all(warmup_hist, rl_hist, neur_eval, out_dir)
    plot_trajectories(observer, navigator, out_dir)

    print(f"\nDone! Outputs in {out_dir}/")
