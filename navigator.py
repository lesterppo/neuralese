"""
Neuralese Blind Navigator — Multi-Agent RL with Emergent Communication
Agent A sees target + Agent B position → outputs 8D Neuralese vector
Agent B receives vector → outputs movement (dx, dy)
Trained jointly with REINFORCE + value baseline
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
TOKENS = 6          # Text baseline uses 6 tokens
MAX_STEPS = 20
EPISODES = 3000
BATCH_EPISODES = 16
LR = 3e-4
GAMMA = 0.95         # Discount factor
ENTROPY_COEF = 0.01  # Encourage exploration
TARGET_THRESH = 0.05

# --- NEURALESE MODELS ---

class NeuraleseObserver(nn.Module):
    """Sees [target_x, target_y, pos_x, pos_y] → outputs 8D Neuralese vector"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT_DIM * 2),  # mean + log_std
        )

    def forward(self, state):
        """Returns: latent_mean, latent_std (for stochastic policy)"""
        out = self.net(state)
        mean = out[:, :LATENT_DIM]
        log_std = out[:, LATENT_DIM:] - 2.0  # Start small noise
        return mean, torch.exp(log_std).clamp(0.01, 1.0)

class NeuraleseNavigator(nn.Module):
    """Receives 8D Neuralese vector → outputs movement (dx, dy) mean + std"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 4),  # dx_mean, dy_mean, dx_log_std, dy_log_std
        )

    def forward(self, latent):
        out = self.net(latent)
        mean = out[:, :2]
        log_std = out[:, 2:] - 1.0  # Action noise
        return mean, torch.exp(log_std).clamp(0.05, 0.5)

# --- TEXT BASELINE MODELS ---

class TextObserver(nn.Module):
    """Sees state → outputs discrete token logits"""
    def __init__(self):
        super().__init__()
        self.tokens = TOKENS
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, TOKENS * VOCAB),
        )

    def forward(self, state):
        logits = self.net(state).view(-1, TOKENS, VOCAB)
        return logits

class TextNavigator(nn.Module):
    """Receives token indices → outputs movement"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.net = nn.Sequential(
            nn.Linear(TOKENS * HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 4),
        )

    def forward(self, tokens_or_soft):
        """tokens_or_soft: [B, TOKENS, VOCAB] soft probs, or [B, TOKENS] int indices"""
        if tokens_or_soft.dim() == 2:
            # Hard tokens
            emb = self.embed(tokens_or_soft).view(-1, TOKENS * HIDDEN)
        else:
            # Soft tokens during training
            emb = torch.matmul(tokens_or_soft, self.embed.weight).view(-1, TOKENS * HIDDEN)
        out = self.net(emb)
        mean = out[:, :2]
        log_std = out[:, 2:] - 1.0
        return mean, torch.exp(log_std).clamp(0.05, 0.5)

# --- ENVIRONMENT ---

def generate_episode_batch(batch_size, observer, navigator, is_text=False, temperature=1.0):
    """Run batch of episodes. Returns: rewards_list, log_probs_list, entropies_list, targets, trajectories, states_list"""
    targets = torch.rand(batch_size, 2) * 2 - 1
    positions = torch.zeros(batch_size, 2)
    active = torch.ones(batch_size, dtype=torch.bool)
    
    all_log_probs = []
    all_entropies = []
    all_rewards = []
    all_states = []
    trajectories = [[] for _ in range(batch_size)]
    
    for b in range(batch_size):
        trajectories[b].append(positions[b].detach().clone().numpy())

    for step in range(MAX_STEPS):
        if not active.any():
            break

        state = torch.cat([targets, positions], dim=1)  # [B, 4]
        all_states.append(state.clone())

        if not is_text:
            latent_mean, latent_std = observer(state)
            latent_dist = D.Normal(latent_mean, latent_std)
            latent = latent_dist.rsample()
            latent_log_prob = latent_dist.log_prob(latent).sum(dim=-1)
            latent_entropy = latent_dist.entropy().sum(dim=-1)

            action_mean, action_std = navigator(latent)
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            action_log_prob = action_dist.log_prob(action).sum(dim=-1)
            action_entropy = action_dist.entropy().sum(dim=-1)

            log_prob = latent_log_prob + action_log_prob
            entropy = latent_entropy + action_entropy
        else:
            logits = observer(state)
            soft = nn.functional.gumbel_softmax(logits.view(-1, VOCAB), tau=temperature, hard=False)
            soft = soft.view(-1, TOKENS, VOCAB)
            tokens = torch.argmax(logits, dim=-1)
            
            cat_dist = D.Categorical(logits=logits.view(-1, VOCAB))
            log_prob = cat_dist.log_prob(tokens.view(-1)).view(-1, TOKENS).sum(dim=-1)
            entropy = cat_dist.entropy().view(-1, TOKENS).sum(dim=-1)

            action_mean, action_std = navigator(soft)
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            action_log_prob = action_dist.log_prob(action).sum(dim=-1)
            action_entropy = action_dist.entropy().sum(dim=-1)
            
            log_prob = log_prob + action_log_prob
            entropy = entropy + action_entropy
        
        new_positions = positions + action
        new_positions = torch.clamp(new_positions, -1, 1)
        
        dists = torch.norm(new_positions - targets, dim=1)
        rewards = -dists
        
        just_finished = (dists < TARGET_THRESH) & active
        active = active & ~just_finished
        
        all_log_probs.append(log_prob)
        all_entropies.append(entropy)
        all_rewards.append(rewards)
        positions = new_positions

        for b in range(batch_size):
            trajectories[b].append(positions[b].detach().clone().numpy())

    return all_rewards, all_log_probs, all_entropies, targets, trajectories, all_states

# --- TRAINING ---

class ValueBaseline(nn.Module):
    """Learns state value for variance reduction"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

def compute_returns(rewards_list, gamma=GAMMA):
    """Compute discounted returns for each step"""
    T = len(rewards_list)
    B = rewards_list[0].shape[0]
    returns = []
    G = torch.zeros(B)
    for t in reversed(range(T)):
        G = rewards_list[t] + gamma * G
        returns.insert(0, G.clone())
    return returns

def train_agent(is_text=False, label="model"):
    if is_text:
        observer = TextObserver()
        navigator = TextNavigator()
    else:
        observer = NeuraleseObserver()
        navigator = NeuraleseNavigator()
    
    baseline = ValueBaseline()
    all_params = list(observer.parameters()) + list(navigator.parameters()) + list(baseline.parameters())
    opt = optim.Adam(all_params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPISODES)
    
    history = {"reward": [], "steps": [], "value_loss": []}
    temp_start, temp_end = 3.0, 0.5
    
    for ep in range(EPISODES):
        temp = temp_start * (temp_end / temp_start) ** (ep / EPISODES) if is_text else 1.0
        
        rewards_list, log_probs_list, entropies_list, targets, trajs, states_list = \
            generate_episode_batch(BATCH_EPISODES, observer, navigator, is_text, temp)
        
        returns_list = compute_returns(rewards_list)
        
        policy_loss = 0.0
        value_loss = 0.0
        n_steps = len(rewards_list)
        
        for t in range(n_steps):
            state_t = states_list[t]
            values = baseline(state_t)
            advantage = (returns_list[t] - values.detach())
            policy_loss = policy_loss - (log_probs_list[t] * advantage).mean()
            value_loss = value_loss + nn.MSELoss()(values, returns_list[t])
        
        # Add entropy bonus
        entropy_bonus = 0.0
        for ent in entropies_list:
            entropy_bonus = entropy_bonus + ent.mean()
        
        total_loss = policy_loss / n_steps + value_loss / n_steps - ENTROPY_COEF * entropy_bonus / n_steps
        
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        scheduler.step()
        
        # Logging
        total_reward = sum(r.sum().item() for r in rewards_list)
        avg_steps = n_steps
        history["reward"].append(total_reward / BATCH_EPISODES)
        history["steps"].append(avg_steps)
        history["value_loss"].append(value_loss.item() / n_steps)
        
        if ep % 500 == 0:
            temp_str = f" temp={temp:.2f}" if is_text else ""
            print(f"  [{label}] Ep {ep:4d}: avg_reward={total_reward/BATCH_EPISODES:.4f} "
                  f"steps={avg_steps} value_loss={value_loss.item()/n_steps:.4f}{temp_str}")
    
    return observer, navigator, baseline, history

# --- EVALUATION ---

def evaluate(observer, navigator, is_text=False, num_eps=100):
    """Run evaluation episodes (no exploration noise)"""
    total_steps = 0
    total_reward = 0.0
    successes = 0
    
    for _ in range(num_eps):
        target = torch.rand(2) * 2 - 1
        pos = torch.zeros(2)
        
        for step in range(MAX_STEPS):
            state = torch.cat([target, pos]).unsqueeze(0)
            
            with torch.no_grad():
                if not is_text:
                    latent_mean, _ = observer(state)
                    action_mean, _ = navigator(latent_mean)
                else:
                    logits = observer(state)
                    tokens = torch.argmax(logits, dim=-1).squeeze(0)
                    action_mean, _ = navigator(tokens.unsqueeze(0))
                
                action = action_mean.squeeze(0)  # No noise
                
            pos = torch.clamp(pos + action, -1, 1)
            dist = torch.norm(pos - target).item()
            
            if dist < TARGET_THRESH:
                successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += MAX_STEPS
        
        total_reward += -dist
    
    return {
        "avg_reward": total_reward / num_eps,
        "avg_steps": total_steps / num_eps,
        "success_rate": successes / num_eps,
    }

# --- VISUALIZATION ---

def plot_trajectories(observer, navigator, is_text, label, out_dir):
    """Plot 5 sample trajectories showing Agent B's path to targets"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i in range(5):
        ax = axes[i]
        target = torch.rand(2) * 2 - 1
        pos = torch.zeros(2)
        path = [pos.clone().numpy()]
        
        for step in range(MAX_STEPS):
            state = torch.cat([target, pos]).unsqueeze(0)
            with torch.no_grad():
                if not is_text:
                    latent_mean, _ = observer(state)
                    action_mean, _ = navigator(latent_mean)
                else:
                    logits = observer(state)
                    tokens = torch.argmax(logits, dim=-1).squeeze(0)
                    action_mean, _ = navigator(tokens.unsqueeze(0))
                action = action_mean.squeeze(0)
            
            pos = torch.clamp(pos + action, -1, 1)
            path.append(pos.clone().numpy())
            if torch.norm(pos - target).item() < TARGET_THRESH:
                break
        
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(path[0, 0], path[0, 1], c='green', s=80, marker='o', label='Start', zorder=5)
        ax.scatter(target[0].item(), target[1].item(), c='red', s=120, marker='*', label='Target', zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=60, marker='s', label='End', zorder=5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Target ({target[0]:.2f}, {target[1]:.2f})")
        if i == 0:
            ax.legend(fontsize=7, loc='lower left')
    
    plt.suptitle(f"{label} — Agent B Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / f"trajectories_{'neuralese' if not is_text else 'text'}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

def plot_comparison(neur_history, text_history, neur_eval, text_eval, out_dir):
    """Side-by-side comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward curves
    ax = axes[0, 0]
    ax.plot(neur_history["reward"], alpha=0.7, label="Neuralese", color="blue")
    ax.plot(text_history["reward"], alpha=0.7, label="Text Baseline", color="red")
    ax.set_title("Average Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps per episode
    ax = axes[0, 1]
    ax.plot(neur_history["steps"], alpha=0.7, label="Neuralese", color="blue")
    ax.plot(text_history["steps"], alpha=0.7, label="Text Baseline", color="red")
    ax.set_title("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar chart: success rate
    ax = axes[1, 0]
    bars = ax.bar(["Neuralese", "Text Baseline"],
                  [neur_eval["success_rate"], text_eval["success_rate"]],
                  color=["blue", "red"], alpha=0.7)
    ax.set_title("Success Rate (reached within 0.05)")
    ax.set_ylabel("Success Rate")
    for bar, val in zip(bars, [neur_eval["success_rate"], text_eval["success_rate"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    # Bar chart: avg steps to target
    ax = axes[1, 1]
    bars = ax.bar(["Neuralese", "Text Baseline"],
                  [neur_eval["avg_steps"], text_eval["avg_steps"]],
                  color=["blue", "red"], alpha=0.7)
    ax.set_title("Average Steps to Target")
    ax.set_ylabel("Steps")
    for bar, val in zip(bars, [neur_eval["avg_steps"], text_eval["avg_steps"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha='center', fontweight='bold')
    
    plt.suptitle("Blind Navigator — Neuralese vs Text Baseline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "navigator_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BLIND NAVIGATOR — Emergent Communication via Neuralese")
    print("=" * 60)
    
    # Train Neuralese
    print("\n[1/2] Training Neuralese Blind Navigator...")
    neur_obs, neur_nav, neur_base, neur_hist = train_agent(is_text=False, label="Neuralese")
    
    # Train Text Baseline
    print("\n[2/2] Training Text Baseline Blind Navigator...")
    text_obs, text_nav, text_base, text_hist = train_agent(is_text=True, label="Text")
    
    # Evaluate
    print("\n" + "=" * 40)
    print("EVALUATION (100 episodes, no exploration noise):")
    neur_eval = evaluate(neur_obs, neur_nav, is_text=False)
    text_eval = evaluate(text_obs, text_nav, is_text=True)
    
    print(f"\n  Neuralese:")
    print(f"    Success Rate:  {neur_eval['success_rate']:.1%}")
    print(f"    Avg Steps:     {neur_eval['avg_steps']:.2f}")
    print(f"    Avg Reward:    {neur_eval['avg_reward']:.4f}")
    
    print(f"\n  Text Baseline:")
    print(f"    Success Rate:  {text_eval['success_rate']:.1%}")
    print(f"    Avg Steps:     {text_eval['avg_steps']:.2f}")
    print(f"    Avg Reward:    {text_eval['avg_reward']:.4f}")
    
    # Bandwidth comparison
    neur_bits = LATENT_DIM * 32
    text_bits = TOKENS * np.log2(VOCAB)
    print(f"\n  Bandwidth: Neuralese = {neur_bits} bits, Text = {text_bits:.0f} bits")
    print(f"  Efficiency: Neuralese uses {neur_bits/text_bits:.1f}× more bits but achieves "
          f"{text_eval['avg_reward']/neur_eval['avg_reward']:.2f}× better reward")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_comparison(neur_hist, text_hist, neur_eval, text_eval, out_dir)
    plot_trajectories(neur_obs, neur_nav, False, "Neuralese", out_dir)
    plot_trajectories(text_obs, text_nav, True, "Text Baseline", out_dir)
    
    print(f"\nDone! Outputs in {out_dir}/")
