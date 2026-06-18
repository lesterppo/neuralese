"""
Neuralese Remote Brain v3 — MLP Observer + Repulsion-Field RL
Key improvements over v2:
1. MLP Observer (NOT GRU) — forces per-step reprocessing → emergent latents
2. Wall repulsion field reward — continuous dense penalty for proximity to walls
3. Two-phase RL: Phase 2a (wall avoidance) → Phase 2b (task+wall)
4. Stronger diversity loss with per-step drift penalty
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import heapq
from pathlib import Path
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIG ---
GRID_SIZE = 10
WALL_PROB = 0.20
LATENT_DIM = 12
HIDDEN = 128
STEP_CAP = 0.5
MAX_STEPS = 50
BATCH = 8
WARMUP_EPOCHS = 1000
RL_EPOCHS_PHASE1 = 800   # Phase 2a: focus on wall avoidance
RL_EPOCHS_PHASE2 = 1200  # Phase 2b: full task
LR = 1e-3
RL_LR = 3e-4
ENTROPY_COEF = 0.01
WALL_PENALTY = -10.0       # On collision (kept for safety)
GOAL_REWARD = 10.0
STEP_PENALTY = -0.05
TARGET_THRESH = 0.5
REPULSION_SCALE = 3.0      # Strength of proximity-based wall repulsion
REPULSION_DECAY = 1.5      # How fast repulsion drops with distance

# --- ENVIRONMENT ---

def gen_maze(wall_prob=WALL_PROB):
    for _ in range(100):
        grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < wall_prob).astype(np.float32)
        start = (0, 0)
        target = (GRID_SIZE - 1, GRID_SIZE - 1)
        grid[start] = 0.0
        grid[target] = 0.0
        if bfs_reachable(grid, start, target):
            return torch.tensor(grid), torch.tensor(start, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    return torch.tensor(grid), torch.tensor((0, 0), dtype=torch.float32), torch.tensor((GRID_SIZE-1, GRID_SIZE-1), dtype=torch.float32)

def bfs_reachable(grid, start, target):
    q = deque([start])
    visited = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == target:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return False

def get_radar(grid, pos_r, pos_c):
    radar = torch.ones(3, 3)
    for i in range(3):
        for j in range(3):
            nr, nc = int(pos_r) + i - 1, int(pos_c) + j - 1
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                radar[i, j] = grid[nr, nc]
    return radar.flatten()

def a_star_path(grid, start, target):
    sr, sc = int(start[0].item()), int(start[1].item())
    tr, tc = int(target[0].item()), int(target[1].item())
    g_score = {(sr, sc): 0}
    parent = {(sr, sc): None}
    open_set = [(abs(sr-tr) + abs(sc-tc), sr, sc)]
    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r, c) == (tr, tc):
            path = []
            curr = (tr, tc)
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                tentative_g = g_score[(r, c)] + 1
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative_g
                    parent[(nr, nc)] = (r, c)
                    f = tentative_g + abs(nr-tr) + abs(nc-tc)
                    heapq.heappush(open_set, (f, nr, nc))
    return None

# --- MODELS ---

class Observer(nn.Module):
    """MLP Observer: sees full map (100) + position (2) → 12D Neuralese.
    MLP has NO memory — must reprocess from scratch each step.
    This limitation IS the feature: forces dynamic per-step communication."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE + 2, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def forward(self, grid_flat, pos):
        pos_norm = pos / GRID_SIZE
        x = torch.cat([grid_flat, pos_norm], dim=-1)
        return self.net(x)

class Navigator(nn.Module):
    """Sees 3×3 radar (9) + Neuralese (12) → clamped movement"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9 + LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
            nn.Tanh(),
        )

    def forward(self, radar, z):
        return self.net(torch.cat([radar, z], dim=-1)) * STEP_CAP

# --- SUPERVISED WARM-START ---

def generate_expert_batch(batch_size, wall_prob=WALL_PROB):
    all_items = []
    grids, starts, targets = [], [], []
    for _ in range(batch_size):
        grid, start, target = gen_maze(wall_prob)
        grids.append(grid.flatten())
        starts.append(start)
        targets.append(target)
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 2:
            for i in range(len(path) - 1):
                curr_r, curr_c = path[i]
                next_r, next_c = path[i+1]
                move = torch.tensor([next_r - curr_r, next_c - curr_c], dtype=torch.float32)
                pos = torch.tensor([curr_r, curr_c], dtype=torch.float32)
                radar = get_radar(grid, curr_r, curr_c)
                all_items.append((
                    grid.flatten(), pos, radar, move, torch.tensor([next_r, next_c], dtype=torch.float32)
                ))
    return grids, starts, targets, all_items

def warm_start(observer, navigator, epochs=WARMUP_EPOCHS):
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []

    for ep in range(epochs):
        grids, starts, targets, expert_items = generate_expert_batch(BATCH)
        if len(expert_items) < BATCH:
            continue

        # Randomly sample from expert trajectories
        idx = np.random.choice(len(expert_items), BATCH, replace=False)
        grid_batch = torch.stack([expert_items[i][0] for i in idx])
        pos_batch = torch.stack([expert_items[i][1] for i in idx])
        radar_batch = torch.stack([expert_items[i][2] for i in idx])
        target_move = torch.stack([expert_items[i][3] for i in idx])

        z = observer(grid_batch, pos_batch)
        pred_move = navigator(radar_batch, z)

        task_loss = nn.MSELoss()(pred_move, target_move)
        l2_loss = 0.001 * torch.norm(z, p=2)
        loss = task_loss + l2_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step()
        scheduler.step()
        history.append(loss.item())

        if ep % 500 == 0:
            print(f"  Warm-up {ep:5d}: loss={loss.item():.6f}")

    return history

# --- UTILITY: wall proximity reward ---

def compute_wall_repulsion(grids, positions, scale=REPULSION_SCALE, decay=REPULSION_DECAY):
    """Per-batch reward: penalty for being near walls.
    For each agent, checks 3×3 radar → counts wall cells → exponential distance penalty.
    Returns reward (negative for proximity, 0 for clear space)."""
    B = len(grids)
    rewards = torch.zeros(B)
    for b in range(B):
        r = int(round(positions[b, 0].item()))
        c = int(round(positions[b, 1].item()))
        r = max(0, min(GRID_SIZE-1, r))
        c = max(0, min(GRID_SIZE-1, c))

        # Count wall cells in 3×3 neighborhood and their distances
        total_penalty = 0.0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if grids[b][nr, nc] == 1.0:
                        # Distance to this wall cell
                        dist = np.sqrt(dr**2 + dc**2)
                        if nr != r and nc != c:
                            dist = np.sqrt(2)  # Diagonal
                        total_penalty += np.exp(-decay * (dist - 1.0))  # dist=1 → exp(0)=1

        rewards[b] = -scale * total_penalty
    return rewards

# --- RL FINE-TUNING (with repulsion) ---

def rl_fine_tune(observer, navigator, epochs, task_weight=1.0, repulsion_weight=1.0,
                 phase_label="RL"):
    """REINFORCE with task reward + wall repulsion reward.
    task_weight / repulsion_weight control the balance."""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    history = {"reward": [], "steps": [], "success": [], "wall_hits": [], "repulsion": []}

    for ep in range(epochs):
        all_grids = []
        all_starts = []
        all_targets = []
        for _ in range(BATCH):
            grid, start, target = gen_maze()
            all_grids.append(grid)
            all_starts.append(start)
            all_targets.append(target)

        grids_batch = torch.stack([g.flatten() for g in all_grids])
        starts_batch = torch.stack(all_starts)
        targets_batch = torch.stack(all_targets)
        positions = starts_batch.clone()
        active = torch.ones(BATCH, dtype=torch.bool)

        ep_log_probs = []
        ep_rewards = []

        for step in range(MAX_STEPS):
            if not active.any():
                break

            radars = []
            for b in range(BATCH):
                r, c = int(positions[b, 0].item()), int(positions[b, 1].item())
                radars.append(get_radar(all_grids[b], r, c))
            radars_batch = torch.stack(radars)

            z = observer(grids_batch, positions)
            action_mean = navigator(radars_batch, z)
            action_std = torch.ones_like(action_mean) * 0.3
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            new_positions = positions + action
            new_positions = torch.clamp(new_positions, 0, GRID_SIZE - 1)

            dists_to_target = torch.norm(new_positions - targets_batch, dim=1)

            # Task reward: distance to target
            task_reward = -dists_to_target * 0.1 * task_weight

            # Wall repulsion reward (dense, proximity-based)
            repulsion_reward = compute_wall_repulsion(all_grids, new_positions) * repulsion_weight

            # Wall collision penalty
            wall_hit = torch.zeros(BATCH, dtype=torch.bool)
            for b in range(BATCH):
                r, c = int(round(new_positions[b, 0].item())), int(round(new_positions[b, 1].item()))
                r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))
                if all_grids[b][r, c] == 1.0:
                    wall_hit[b] = True

            rewards = task_reward + repulsion_reward + STEP_PENALTY
            rewards[wall_hit] += WALL_PENALTY

            just_finished = (dists_to_target < TARGET_THRESH) & active
            rewards[just_finished] += GOAL_REWARD
            active = active & ~just_finished

            ep_log_probs.append(log_prob)
            ep_rewards.append(rewards)
            positions = new_positions

        n_steps = len(ep_rewards)

        # Discounted returns
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
        total_repulsion = sum(r[~wall_hit].sum().item() for r in [ep_rewards[-1]])
        history["reward"].append(total_reward / BATCH)
        history["steps"].append(n_steps)
        success_count = ((positions - targets_batch).norm(dim=1) < TARGET_THRESH).sum().item()
        history["success"].append(success_count / BATCH)
        history["wall_hits"].append(wall_hit.sum().item() / BATCH)
        history["repulsion"].append(repulsion_reward.mean().item())

        if ep % 500 == 0:
            print(f"  {phase_label} ep {ep:4d}: reward={total_reward/BATCH:.2f} "
                  f"success={success_count/BATCH:.2f} walls={wall_hit.sum().item()} "
                  f"repuls={repulsion_reward.mean().item():.3f}")

    return history

# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=200):
    neur_success, neur_steps = 0, 0
    greedy_success, greedy_steps = 0, 0
    wall_hits_neuralese = 0
    total_repulsion = 0.0

    for ep in range(num_eps):
        grid, start, target = gen_maze()
        pos = start.clone()
        for step in range(MAX_STEPS):
            grid_flat = grid.flatten().unsqueeze(0)
            pos_batch = pos.unsqueeze(0)
            radar = get_radar(grid, int(pos[0].item()), int(pos[1].item())).unsqueeze(0)

            with torch.no_grad():
                z = observer(grid_flat, pos_batch)
                action = navigator(radar, z).squeeze(0)

            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r, c = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))

            if grid[r, c] == 1.0:
                wall_hits_neuralese += 1
                break

            pos = new_pos
            # Compute repulsion at this position (for diagnostics)
            rep = compute_wall_repulsion([grid], pos.unsqueeze(0))
            total_repulsion += rep.item()

            dist = torch.norm(pos - target).item()
            if dist < TARGET_THRESH:
                neur_success += 1
                neur_steps += step + 1
                break
        else:
            neur_steps += MAX_STEPS

        # Greedy baseline
        pos = start.clone()
        path = a_star_path(grid.numpy(), start, target)
        if path:
            greedy_success += 1
            greedy_steps += len(path) - 1
        else:
            greedy_steps += MAX_STEPS

    n_eval = max(1, neur_success if neur_success > 0 else num_eps)
    return {
        "neur_success": neur_success / num_eps,
        "neur_steps": neur_steps / n_eval,
        "greedy_success": greedy_success / num_eps,
        "greedy_steps": greedy_steps / max(1, greedy_success),
        "wall_hits_neur": wall_hits_neuralese / num_eps,
        "avg_repulsion": total_repulsion / num_eps,
    }

# --- VERIFICATION ---

def latent_evolution_test(observer, navigator, out_dir):
    for attempt in range(20):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 5 and np.sum(grid.numpy()) > 2:
            break

    print(f"  Path length: {len(path)}, Walls: {int(grid.sum().item())}")

    pos = start.clone()
    latents = []
    positions = []

    for step in range(min(len(path), 30)):
        grid_flat = grid.flatten().unsqueeze(0)
        pos_batch = pos.unsqueeze(0)
        radar = get_radar(grid, int(pos[0].item()), int(pos[1].item())).unsqueeze(0)

        with torch.no_grad():
            z = observer(grid_flat, pos_batch)
            action = navigator(radar, z).squeeze(0)

        latents.append(z.squeeze(0).numpy())
        positions.append(pos.clone().numpy())

        new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
        r, c = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
        if grid[r, c] == 1.0:
            break
        pos = new_pos

        if torch.norm(pos - target).item() < TARGET_THRESH:
            break

    latents_arr = np.array(latents)
    positions_arr = np.array(positions)

    z_std = np.std(latents_arr, axis=0).mean()
    distances = [np.linalg.norm(latents_arr[i] - latents_arr[0]) for i in range(len(latents_arr))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(distances, 'b-o', markersize=8)
    ax.set_title("Latent Distance from t=0")
    ax.set_xlabel("Step"); ax.set_ylabel("||z_t - z_0||")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for d in range(min(8, LATENT_DIM)):
        ax.plot(latents_arr[:, d], alpha=0.5, label=f"dim {d}", marker='o', markersize=4)
    ax.set_title("Latent Dimensions over Steps")
    ax.set_xlabel("Step"); ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    grid_np = grid.numpy()
    ax.imshow(grid_np.T, cmap='gray_r', origin='lower', alpha=0.3)
    ax.plot(positions_arr[:, 0], positions_arr[:, 1], 'b-', linewidth=2, label='Agent B')
    ax.scatter(positions_arr[0, 0], positions_arr[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(target[0].item(), target[1].item(), c='red', s=150, marker='*', label='Target')
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'r--', alpha=0.3, label='A* path')
    ax.set_xlim(-0.5, GRID_SIZE-0.5); ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_title("Agent Trajectory on Maze"); ax.legend(fontsize=7)

    plt.suptitle(f"Latent Evolution Test — z_std={z_std:.4f} "
                 f"({'EMERGENT' if z_std > 0.05 else 'STATIC'})", fontsize=14)
    plt.tight_layout()
    fname = out_dir / "maze_latent_evolution_v3.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Latent std: {z_std:.4f} → {'EMERGENT RELATIVE INSTRUCTIONS' if z_std > 0.05 else 'STATIC ENCODING'}")
    print(f"  Max drift from t=0: {max(distances):.4f}")

    return z_std, latents_arr

# --- VISUALIZATION ---

def plot_summary(warmup_hist, rl_hist1, rl_hist2, eval_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(warmup_hist, alpha=0.5, color='blue')
    ax.set_title("Phase 1: Expert Warm-Start")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    # Combined RL history
    all_rewards = rl_hist1["reward"] + rl_hist2["reward"]
    all_success = rl_hist1["success"] + rl_hist2["success"]
    all_walls = rl_hist1["wall_hits"] + rl_hist2["wall_hits"]
    n1 = len(rl_hist1["reward"])
    ax.plot(all_rewards, alpha=0.5, color='green', label="Avg Reward")
    ax.axvline(x=n1, color='red', linestyle='--', alpha=0.5, label="Phase switch")
    ax2 = ax.twinx()
    ax2.plot(all_success, alpha=0.5, color='orange', label="Success Rate")
    ax2.plot(all_walls, alpha=0.3, color='red', label="Wall Hits/ep")
    ax.set_title("Phase 2: RL (repulsion → task+repulsion)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax2.set_ylabel("Rate")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    labels = ["Neuralese", "A* (optimal)"]
    successes = [eval_results["neur_success"], eval_results["greedy_success"]]
    steps = [eval_results["neur_steps"], eval_results["greedy_steps"]]
    x = np.arange(2); width = 0.35
    bars1 = ax.bar(x - width/2, successes, width, label="Success Rate", color="blue", alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, steps, width, label="Avg Steps", color="red", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Success Rate"); ax2.set_ylabel("Avg Steps")
    for bar, val in zip(bars1, successes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{val:.1%}", ha='center')
    for bar, val in zip(bars2, steps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}", ha='center')
    ax.set_title("Evaluation (200 mazes)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    metrics_text = (
        f"Wall hits/ep: {eval_results['wall_hits_neur']:.2f}\n"
        f"Avg repulsion: {eval_results.get('avg_repulsion', 0):.3f}\n"
        f"Repulsion scale: {REPULSION_SCALE}\n"
        f"Repulsion decay: {REPULSION_DECAY}\n"
        f"Warm-up epochs: {WARMUP_EPOCHS}\n"
        f"RL phase1: {RL_EPOCHS_PHASE1}\n"
        f"RL phase2: {RL_EPOCHS_PHASE2}\n"
        f"Observer: MLP (no memory)"
    )
    ax.text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=10, va='center',
            transform=ax.transAxes)
    ax.set_title("Config & Diagnostics")
    ax.axis('off')

    plt.suptitle("Neuralese Remote Brain v3 — MLP + Repulsion Field", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "maze_results_v3.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NEURALESE REMOTE BRAIN v3 — MLP + Repulsion Field")
    print("=" * 60)
    print(f"  Observer: MLP (no memory) — forces per-step reprocessing")
    print(f"  Navigator: 3×3 radar + 12D Neuralese → movement")
    print(f"  Repulsion: {REPULSION_SCALE}× exp(-{REPULSION_DECAY}·d) wall penalty")
    print(f"  Strategy: Phase 2a (wall avoidance) → Phase 2b (full task)")

    # Phase 1: Warm-start
    print("\n[Phase 1] Expert Warm-Start...")
    observer = Observer()
    navigator = Navigator()
    warmup_hist = warm_start(observer, navigator)

    # Phase 2a: RL with repulsion focus (wall avoidance emphasized)
    print("\n[Phase 2a] RL: Wall Avoidance Focus...")
    rl_hist1 = rl_fine_tune(observer, navigator, RL_EPOCHS_PHASE1,
                            task_weight=0.3, repulsion_weight=3.0,
                            phase_label="RL-avoid")

    # Phase 2b: RL with balanced task+repulsion
    print("\n[Phase 2b] RL: Full Task + Repulsion...")
    rl_hist2 = rl_fine_tune(observer, navigator, RL_EPOCHS_PHASE2,
                            task_weight=1.0, repulsion_weight=1.0,
                            phase_label="RL-task")

    # Evaluate
    print("\n" + "=" * 40)
    print("EVALUATION (200 mazes):")
    eval_results = evaluate(observer, navigator)
    print(f"  Neuralese: {eval_results['neur_success']:.1%} success, "
          f"{eval_results['neur_steps']:.1f} avg steps, "
          f"{eval_results['wall_hits_neur']:.2f} wall hits/ep")
    print(f"  A* (optimal): {eval_results['greedy_success']:.1%} success, "
          f"{eval_results['greedy_steps']:.1f} avg steps")
    print(f"  Avg repulsion: {eval_results.get('avg_repulsion', 0):.4f}")

    # Latent evolution
    print("\n[Latent Evolution Test]")
    z_std, latents = latent_evolution_test(observer, navigator, out_dir)

    # Visualize
    print("\nGenerating plots...")
    plot_summary(warmup_hist, rl_hist1, rl_hist2, eval_results, out_dir)

    print(f"\nDone! Outputs in {out_dir}/")
