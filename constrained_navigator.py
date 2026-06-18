"""
Neuralese Constrained Navigator — Step-Capped with Obstacle Avoidance
Forces multi-step relative-instruction communication (not one-shot teleport)
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
from sklearn.decomposition import PCA

# --- CONFIG ---
HIDDEN = 128
LATENT_DIM = 8
STEP_CAP = 0.15
MAX_STEPS = 30
WARMUP_STEPS = 3000
RL_EPISODES = 3000
BATCH = 64
LR = 1e-3
RL_LR = 3e-4
ENTROPY_COEF = 0.01
TARGET_THRESH = 0.05
OBSTACLE_PENALTY = -2.0
OBSTACLE_RADIUS = 0.25

# --- MODELS ---

class Observer(nn.Module):
    """Sees [target_x, target_y, pos_x, pos_y, wall_x, wall_y] → 8D Neuralese"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def forward(self, state):
        return self.net(state)

class Navigator(nn.Module):
    """8D Neuralese → clamped movement [-STEP_CAP, STEP_CAP]"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z) * STEP_CAP

# --- WARM-START ---

def warm_start(observer, navigator, steps=WARMUP_STEPS):
    """Pre-train: Observer encodes state → Navigator outputs clamped movement toward target"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    history = []

    for step in range(steps):
        targets = torch.rand(BATCH, 2) * 2 - 1
        positions = torch.rand(BATCH, 2) * 2 - 1
        # Random obstacle positions (or None)
        has_wall = torch.rand(BATCH) < 0.5
        walls = torch.rand(BATCH, 2) * 2 - 1
        walls[~has_wall] = 999.0  # Far away = no wall
        
        state = torch.cat([targets, positions, walls], dim=1)
        
        # Target: clamped displacement toward target, but avoiding wall
        raw_diff = targets - positions
        diff_norm = torch.norm(raw_diff, dim=1, keepdim=True).clamp(min=1e-8)
        direction = raw_diff / diff_norm
        
        # Simple obstacle avoidance in warm-start: push away from wall if close
        to_wall = walls - positions
        wall_dist = torch.norm(to_wall, dim=1, keepdim=True).clamp(min=1e-8)
        wall_dir = to_wall / wall_dist
        repulsion = -wall_dir * torch.exp(-wall_dist * 5.0) * 0.1
        
        target_movement = direction * STEP_CAP + repulsion
        target_movement = torch.clamp(target_movement, -STEP_CAP, STEP_CAP)

        latent = observer(state)
        movement = navigator(latent)
        loss = nn.MSELoss()(movement, target_movement) + 0.001 * torch.norm(latent, p=2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history.append(loss.item())

        if step % 500 == 0:
            print(f"  Warm-up {step:5d}: loss={loss.item():.6f}")

    return history

# --- RL FINE-TUNING ---

def rl_fine_tune(observer, navigator, episodes=RL_EPISODES):
    """REINFORCE: multi-step navigation with obstacles, step cap"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    history = {"reward": [], "steps": [], "success": []}

    for ep in range(episodes):
        targets = torch.rand(BATCH, 2) * 2 - 1
        positions = torch.zeros(BATCH, 2)
        has_wall = torch.rand(BATCH) < 0.5
        walls = torch.rand(BATCH, 2) * 2 - 1
        walls[~has_wall] = 999.0

        active = torch.ones(BATCH, dtype=torch.bool)
        ep_log_probs = []
        ep_rewards = []

        for step in range(MAX_STEPS):
            if not active.any():
                break

            state = torch.cat([targets, positions, walls], dim=1)
            latent = observer(state)

            action_mean = navigator(latent)
            action_std = torch.ones_like(action_mean) * 0.05
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            new_positions = positions + action
            new_positions = torch.clamp(new_positions, -1, 1)

            dists_to_target = torch.norm(new_positions - targets, dim=1)
            dists_to_wall = torch.norm(new_positions - walls, dim=1)
            
            rewards = -dists_to_target
            # Obstacle penalty
            wall_hit = (dists_to_wall < OBSTACLE_RADIUS) & has_wall
            rewards[wall_hit] += OBSTACLE_PENALTY

            just_finished = (dists_to_target < TARGET_THRESH) & active
            active = active & ~just_finished

            ep_log_probs.append(log_prob)
            ep_rewards.append(rewards)
            positions = new_positions

        n_steps = len(ep_rewards)
        
        returns = []
        G = torch.zeros(BATCH)
        gamma = 0.95
        for t in reversed(range(n_steps)):
            G = ep_rewards[t] + gamma * G
            returns.insert(0, G.clone())

        policy_loss = 0.0
        for t in range(n_steps):
            advantage = returns[t] - returns[t].mean()
            policy_loss = policy_loss - (ep_log_probs[t] * advantage.detach()).mean()
        policy_loss = policy_loss / n_steps

        opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step()

        total_reward = sum(r.sum().item() for r in ep_rewards)
        history["reward"].append(total_reward / BATCH)
        history["steps"].append(n_steps)
        successes = ((positions - targets).norm(dim=1) < TARGET_THRESH).sum().item()
        history["success"].append(successes / BATCH)

        if ep % 500 == 0:
            print(f"  RL ep {ep:4d}: reward={total_reward/BATCH:.3f} "
                  f"steps={n_steps} success={successes/BATCH:.2f}")

    return history

# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=200):
    """Evaluate Neuralese and Greedy baseline"""
    neur_steps = 0
    neur_success = 0
    greedy_steps = 0
    greedy_success = 0

    for _ in range(num_eps):
        target = torch.rand(2) * 2 - 1
        has_wall = torch.rand(1).item() < 0.5
        wall = torch.rand(2) * 2 - 1 if has_wall else torch.tensor([999.0, 999.0])
        
        # Neuralese
        pos = torch.zeros(2)
        for step in range(MAX_STEPS):
            state = torch.cat([target, pos, wall]).unsqueeze(0)
            with torch.no_grad():
                latent = observer(state)
                action = navigator(latent).squeeze(0)
            pos = torch.clamp(pos + action, -1, 1)
            if torch.norm(pos - target).item() < TARGET_THRESH:
                neur_success += 1
                neur_steps += step + 1
                break
        else:
            neur_steps += MAX_STEPS

        # Greedy baseline
        pos = torch.zeros(2)
        for step in range(MAX_STEPS):
            raw = target - pos
            dist = torch.norm(raw).item()
            if dist < TARGET_THRESH:
                greedy_success += 1
                greedy_steps += step + 1
                break
            direction = raw / max(dist, 1e-8)
            # Simple wall avoidance
            if has_wall:
                to_wall = wall - pos
                wall_dist = torch.norm(to_wall).item()
                if wall_dist < OBSTACLE_RADIUS + 0.1:
                    push_away = -to_wall / max(wall_dist, 1e-8) * 0.15
                    action = direction * STEP_CAP * 0.5 + push_away
                else:
                    action = direction * STEP_CAP
            else:
                action = direction * STEP_CAP
            pos = torch.clamp(pos + action, -1, 1)
        else:
            greedy_steps += MAX_STEPS

    return {
        "neur_success": neur_success / num_eps,
        "neur_steps": neur_steps / num_eps,
        "greedy_success": greedy_success / num_eps,
        "greedy_steps": greedy_steps / num_eps,
    }

# --- VERIFICATION: Latent Evolution Test ---

def latent_evolution_test(observer, navigator, out_dir):
    """Test if Neuralese encodes RELATIVE instructions (z changes per step)"""
    target = torch.tensor([0.8, -0.5])
    pos = torch.zeros(2)
    wall = torch.tensor([999.0, 999.0])  # No wall for clean test
    
    latents = []
    positions = []
    
    for step in range(20):
        state = torch.cat([target, pos, wall]).unsqueeze(0)
        with torch.no_grad():
            z = observer(state).squeeze(0).numpy()
            action = navigator(torch.tensor(z).unsqueeze(0)).squeeze(0)
        latents.append(z)
        positions.append(pos.clone().numpy())
        pos = torch.clamp(pos + action, -1, 1)
        if torch.norm(pos - target).item() < TARGET_THRESH:
            break
    
    latents = np.array(latents)
    positions = np.array(positions)
    
    # PCA of latent vectors
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # t-SNE-like PCA plot
    ax = axes[0]
    sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=range(len(z_pca)), cmap="viridis", s=80)
    for i in range(len(z_pca)):
        ax.annotate(str(i), (z_pca[i, 0], z_pca[i, 1]), fontsize=8, ha='right')
    ax.plot(z_pca[:, 0], z_pca[:, 1], 'k--', alpha=0.3)
    ax.set_title("Latent Vector Evolution (PCA)\nDots = z_t at each step")
    plt.colorbar(sc, ax=ax, label="Step")
    
    # Position trajectory
    ax = axes[1]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(target[0].item(), target[1].item(), c='red', s=150, marker='*', label='Target')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title("Agent B Trajectory"); ax.legend()
    
    # Latent vector values over time
    ax = axes[2]
    for d in range(min(8, LATENT_DIM)):
        ax.plot(latents[:, d], alpha=0.5, label=f"dim {d}")
    ax.set_title("Latent Dimensions over Steps")
    ax.set_xlabel("Step"); ax.set_ylabel("Value")
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Latent Evolution Test: Does z_t change per step?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "latent_evolution.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")
    
    # Quantitative test: if z is static, std across steps ≈ 0
    z_std = np.std(latents, axis=0).mean()
    print(f"  Latent variation (mean std across dims): {z_std:.4f}")
    print(f"  {'EMERGENT RELATIVE INSTRUCTIONS' if z_std > 0.1 else 'STATIC ENCODING (lookup table)'}")

# --- VISUALIZATION ---

def plot_all(warmup_hist, rl_hist, eval_results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(warmup_hist, alpha=0.5, color='blue')
    ax.set_title("Phase 1: Warm-Start (Clamped Movement)")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(rl_hist["reward"], alpha=0.7, color='green', label="Reward")
    ax2 = ax.twinx()
    ax2.plot(rl_hist["success"], alpha=0.5, color='orange', label="Success")
    ax.set_title("Phase 2: RL Navigation")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward"); ax2.set_ylabel("Success Rate")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [eval_results["neur_success"], eval_results["greedy_success"]],
           width, label="Neuralese", color="blue", alpha=0.7)
    ax.bar(x + width/2, [eval_results["neur_steps"]/MAX_STEPS, eval_results["greedy_steps"]/MAX_STEPS],
           width, label="Greedy", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Success Rate", "Steps (norm)"])
    ax.set_title("Neuralese vs Greedy Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Neuralese Constrained Navigator", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "constrained_results.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NEURALESE CONSTRAINED NAVIGATOR — Obstacle Avoidance")
    print("=" * 60)
    
    # Phase 1: Warm-start
    print("\n[Phase 1] Warm-Start (clamped movement prediction)...")
    observer = Observer()
    navigator = Navigator()
    warmup_hist = warm_start(observer, navigator)
    
    # Phase 2: RL
    print("\n[Phase 2] RL Fine-Tuning...")
    rl_hist = rl_fine_tune(observer, navigator)
    
    # Evaluate
    print("\n" + "=" * 40)
    print("EVALUATION (200 episodes):")
    eval_results = evaluate(observer, navigator)
    print(f"  Neuralese:  {eval_results['neur_success']:.1%} success, {eval_results['neur_steps']:.2f} avg steps")
    print(f"  Greedy:     {eval_results['greedy_success']:.1%} success, {eval_results['greedy_steps']:.2f} avg steps")
    
    # Latent evolution test
    print("\n[Latent Evolution Test]")
    latent_evolution_test(observer, navigator, out_dir)
    
    # Visualize
    print("\nGenerating plots...")
    plot_all(warmup_hist, rl_hist, eval_results, out_dir)
    
    print(f"\nDone! Outputs in {out_dir}/")
