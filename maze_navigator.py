"""
Neuralese Remote Brain — Blind Navigator on 10×10 Maze
Observer sees full map → compresses path into 8D Neuralese
Navigator sees only 3×3 radar + Neuralese → outputs movement
Greedy baseline fails (hits walls); Neuralese succeeds via remote path planning
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
LATENT_DIM = 12  # 8→12: more headroom for spatial + navigational encoding
HIDDEN = 128
STEP_CAP = 0.5  # Grid cells are at integer coordinates; step=1 moves to adjacent cell
MAX_STEPS = 50
BATCH = 8
WARMUP_EPOCHS = 500
RL_EPOCHS = 500
LR = 1e-3
RL_LR = 3e-4
ENTROPY_COEF = 0.01
WALL_PENALTY = -10.0  # Stronger penalty for hitting walls
GOAL_REWARD = 10.0
STEP_PENALTY = -0.1
TARGET_THRESH = 0.5

# --- ENVIRONMENT ---

def gen_maze(wall_prob=WALL_PROB):
    """Generate 10x10 binary grid, start, target. Ensure path exists."""
    for _ in range(100):
        grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < wall_prob).astype(np.float32)
        start = (0, 0)
        target = (GRID_SIZE - 1, GRID_SIZE - 1)
        grid[start] = 0.0
        grid[target] = 0.0
        
        # BFS to check reachability
        if bfs_reachable(grid, start, target):
            return torch.tensor(grid), torch.tensor(start, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    # Fallback: empty grid
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
    """3×3 local view around position"""
    radar = torch.ones(3, 3)
    for i in range(3):
        for j in range(3):
            nr, nc = int(pos_r) + i - 1, int(pos_c) + j - 1
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                radar[i, j] = grid[nr, nc]
    return radar.flatten()

def a_star_path(grid, start, target):
    """A* returns list of (r,c) positions from start to target"""
    sr, sc = int(start[0].item()), int(start[1].item())
    tr, tc = int(target[0].item()), int(target[1].item())
    
    g_score = {(sr, sc): 0}
    parent = {(sr, sc): None}
    open_set = [(abs(sr-tr) + abs(sc-tc), sr, sc)]
    
    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r, c) == (tr, tc):
            # Reconstruct path
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
    """Recurrent Observer: GRU maintains state across navigation steps.
    Input: grid (100) + position (2) + previous hidden → Output: updated Neuralese (12D)"""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE + 2, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.gru = nn.GRUCell(HIDDEN, latent_dim)
        # Orthogonal init for GRU stability over long sequences
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, grid_flat, pos, hidden=None):
        # Normalize pos to [0, 1] for GRU stability
        pos_norm = pos / GRID_SIZE
        x = torch.cat([grid_flat, pos_norm], dim=-1)
        encoded = self.encoder(x)
        if hidden is None:
            hidden = torch.zeros(grid_flat.size(0), self.latent_dim, device=grid_flat.device)
        z = self.gru(encoded, hidden)
        return z

class Navigator(nn.Module):
    """Sees 3×3 radar (9) + Neuralese (8) → movement [-STEP_CAP, STEP_CAP]"""
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
    """Generate batch of mazes with expert (A*) trajectories"""
    grids, starts, targets = [], [], []
    expert_actions = []  # (state, next_move) pairs
    
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
                # State: [grid_flat, pos, radar, next_move]
                radar = get_radar(grid, curr_r, curr_c)
                expert_actions.append((
                    grid.flatten(), pos, radar, move, torch.tensor([next_r, next_c], dtype=torch.float32)
                ))
    
    return grids, starts, targets, expert_actions

def warm_start(observer, navigator, epochs=WARMUP_EPOCHS):
    """Pre-train using A* trajectories — sequential through GRU"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []

    for ep in range(epochs):
        # Generate fresh batch of sequential paths each epoch
        grids, starts, targets, _ = generate_expert_batch(BATCH)
        
        total_loss = 0.0
        hidden = torch.zeros(BATCH, LATENT_DIM)
        positions = torch.stack(starts)
        prev_z = None
        
        # Process up to MAX_STEPS along A* paths
        for step in range(MAX_STEPS):
            # Compute expert move for each agent
            expert_moves = []
            radars = []
            active_mask = torch.zeros(BATCH, dtype=torch.bool)
            
            for b in range(BATCH):
                r, c = int(positions[b, 0].item()), int(positions[b, 1].item())
                # Get A* next step (pass as tuples, not tensors)
                path = a_star_path(grids[b].numpy().reshape(GRID_SIZE, GRID_SIZE),
                                   torch.tensor([r, c], dtype=torch.float32),
                                   targets[b])
                if path and len(path) >= 2:
                    next_r, next_c = path[1]
                    expert_moves.append([next_r - r, next_c - c])
                    active_mask[b] = True
                else:
                    expert_moves.append([0.0, 0.0])
                
                radars.append(get_radar(grids[b].reshape(GRID_SIZE, GRID_SIZE), r, c))
            
            if not active_mask.any():
                break
            
            grids_batch = torch.stack(grids)
            target_move = torch.tensor(expert_moves, dtype=torch.float32)
            radars_batch = torch.stack(radars)
            
            # GRU forward: hidden state carries across steps
            z = observer(grids_batch, positions, hidden)
            pred_move = navigator(radars_batch, z)
            
            # Task loss (only on active agents)
            task_loss = nn.MSELoss()(pred_move[active_mask], target_move[active_mask])
            
            # Diversity loss (scheduled decay)
            div_weight = 0.05 * max(0.0, 1.0 - ep / WARMUP_EPOCHS)
            if prev_z is not None:
                diversity_loss = -torch.norm(z - prev_z.detach(), p=2, dim=1).mean() * div_weight
            else:
                diversity_loss = 0.0
            
            l2_loss = 0.001 * torch.norm(z, p=2)
            
            step_loss = task_loss + diversity_loss + l2_loss
            total_loss = total_loss + step_loss
            
            # Update state
            hidden = z
            prev_z = z
            positions[active_mask] = positions[active_mask] + target_move[active_mask]
            positions = torch.clamp(positions, 0, GRID_SIZE - 1)
        
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step()
        scheduler.step()
        history.append(total_loss.item() / max(1, step + 1))
        
        if ep % 500 == 0:
            print(f"  Warm-up {ep:5d}: loss={total_loss.item()/max(1,step+1):.4f}")

    return history

# --- RL FINE-TUNING ---

def rl_fine_tune(observer, navigator, epochs=RL_EPOCHS):
    """REINFORCE: multi-step navigation on random mazes, wall penalty"""
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    history = {"reward": [], "steps": [], "success": []}
    
    for ep in range(epochs):
        # Generate batch of mazes
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
        hidden = torch.zeros(BATCH, LATENT_DIM)  # GRU hidden state
        
        ep_log_probs = []
        ep_rewards = []
        
        for step in range(MAX_STEPS):
            if not active.any():
                break
            
            # Radar for each agent
            radars = []
            for b in range(BATCH):
                r, c = int(positions[b, 0].item()), int(positions[b, 1].item())
                radars.append(get_radar(all_grids[b], r, c))
            radars_batch = torch.stack(radars)
            
            z = observer(grids_batch, positions, hidden)
            hidden = z  # Carry GRU state forward

            action_mean = navigator(radars_batch, z)
            action_std = torch.ones_like(action_mean) * 0.3
            action_dist = D.Normal(action_mean, action_std)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            new_positions = positions + action
            new_positions = torch.clamp(new_positions, 0, GRID_SIZE - 1)
            
            # Compute rewards
            dists_to_target = torch.norm(new_positions - targets_batch, dim=1)
            
            # Wall check
            wall_hit = torch.zeros(BATCH, dtype=torch.bool)
            for b in range(BATCH):
                r, c = int(round(new_positions[b, 0].item())), int(round(new_positions[b, 1].item()))
                r = max(0, min(GRID_SIZE-1, r))
                c = max(0, min(GRID_SIZE-1, c))
                if all_grids[b][r, c] == 1.0:
                    wall_hit[b] = True
            
            rewards = -dists_to_target * 0.1 + STEP_PENALTY
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
        history["reward"].append(total_reward / BATCH)
        history["steps"].append(n_steps)
        success_count = ((positions - targets_batch).norm(dim=1) < TARGET_THRESH).sum().item()
        history["success"].append(success_count / BATCH)
        
        if ep % 500 == 0:
            print(f"  RL ep {ep:4d}: reward={total_reward/BATCH:.2f} "
                  f"steps={n_steps} success={success_count/BATCH:.2f}")
    
    return history

# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=200):
    """Evaluate on mazes with Neuralese vs Greedy"""
    neur_success, neur_steps = 0, 0
    greedy_success, greedy_steps = 0, 0
    wall_hits_neuralese, wall_hits_greedy = 0, 0
    
    for ep in range(num_eps):
        grid, start, target = gen_maze()
        
        # Neuralese
        pos = start.clone()
        hidden = torch.zeros(1, LATENT_DIM)
        for step in range(MAX_STEPS):
            grid_flat = grid.flatten().unsqueeze(0)
            pos_batch = pos.unsqueeze(0)
            radar = get_radar(grid, int(pos[0].item()), int(pos[1].item())).unsqueeze(0)
            
            with torch.no_grad():
                z = observer(grid_flat, pos_batch, hidden)
                hidden = z
                action = navigator(radar, z).squeeze(0)
            
            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r, c = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))
            
            if grid[r, c] == 1.0:  # Wall hit
                wall_hits_neuralese += 1
                break
            
            pos = new_pos
            dist = torch.norm(pos - target).item()
            if dist < TARGET_THRESH:
                neur_success += 1
                neur_steps += step + 1
                break
        else:
            neur_steps += MAX_STEPS
        
        # Greedy baseline (A*-based, always optimal for static mazes)
        pos = start.clone()
        path = a_star_path(grid.numpy(), start, target)
        if path:
            greedy_success += 1
            greedy_steps += len(path) - 1
        else:
            greedy_steps += MAX_STEPS
    
    return {
        "neur_success": neur_success / num_eps,
        "neur_steps": neur_steps / max(1, neur_success),
        "greedy_success": greedy_success / num_eps,
        "greedy_steps": greedy_steps / max(1, greedy_success),
        "wall_hits_neur": wall_hits_neuralese / num_eps,
    }

# --- VERIFICATION ---

def latent_evolution_test(observer, navigator, out_dir):
    """Test if z changes during navigation around walls"""
    for attempt in range(20):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 5 and np.sum(grid.numpy()) > 2:  # Has walls
            break
    
    print(f"  Path length: {len(path)}, Walls: {int(grid.sum().item())}")
    
    pos = start.clone()
    latents = []
    positions = []
    hidden = torch.zeros(1, LATENT_DIM)
    
    for step in range(min(len(path), 30)):
        grid_flat = grid.flatten().unsqueeze(0)
        pos_batch = pos.unsqueeze(0)
        radar = get_radar(grid, int(pos[0].item()), int(pos[1].item())).unsqueeze(0)
        
        with torch.no_grad():
            z = observer(grid_flat, pos_batch, hidden)
            hidden = z
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
    
    latents = np.array(latents)
    positions = np.array(positions)
    
    z_std = np.std(latents, axis=0).mean()
    distances = [np.linalg.norm(latents[i] - latents[0]) for i in range(len(latents))]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Latent drift from initial
    ax = axes[0]
    ax.plot(distances, 'b-o', markersize=8)
    ax.set_title("Latent Distance from t=0")
    ax.set_xlabel("Step"); ax.set_ylabel("||z_t - z_0||")
    ax.grid(True, alpha=0.3)
    
    # Latent dimensions
    ax = axes[1]
    for d in range(min(8, LATENT_DIM)):
        ax.plot(latents[:, d], alpha=0.5, label=f"dim {d}", marker='o', markersize=4)
    ax.set_title("Latent Dimensions over Steps")
    ax.set_xlabel("Step"); ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    # Maze visualization with path
    ax = axes[2]
    grid_np = grid.numpy()
    ax.imshow(grid_np.T, cmap='gray_r', origin='lower', alpha=0.3)
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Agent B')
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(target[0].item(), target[1].item(), c='red', s=150, marker='*', label='Target')
    # Plot A* path for reference
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'r--', alpha=0.3, label='A* path')
    ax.set_xlim(-0.5, GRID_SIZE-0.5); ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_title("Agent Trajectory on Maze"); ax.legend(fontsize=7)
    
    plt.suptitle(f"Latent Evolution Test — z_std={z_std:.4f} "
                 f"({'EMERGENT' if z_std > 0.05 else 'STATIC'})", fontsize=14)
    plt.tight_layout()
    fname = out_dir / "maze_latent_evolution.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Latent std: {z_std:.4f} → {'EMERGENT RELATIVE INSTRUCTIONS' if z_std > 0.05 else 'STATIC ENCODING'}")
    print(f"  Max drift from t=0: {max(distances):.4f}")

# --- VISUALIZATION ---

def plot_summary(warmup_hist, rl_hist, eval_results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(warmup_hist, alpha=0.5, color='blue')
    ax.set_title("Phase 1: Expert Warm-Start")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(rl_hist["reward"], alpha=0.7, color='green', label="Reward")
    ax2 = ax.twinx()
    ax2.plot(rl_hist["success"], alpha=0.5, color='orange', label="Success")
    ax.set_title("Phase 2: RL Navigation")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax2.set_ylabel("Success")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
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
    
    plt.suptitle("Neuralese Remote Brain — Maze Navigation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "maze_results.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NEURALESE REMOTE BRAIN — Blind Navigator on 10×10 Maze")
    print("=" * 60)
    print(f"  Observer: sees full map (100) + position (2) → {LATENT_DIM}D Neuralese")
    print(f"  Navigator: sees 3×3 radar (9) + {LATENT_DIM}D Neuralese → movement")
    print(f"  Key: Navigator CANNOT see target or full map — must trust Observer")
    
    # Phase 1
    print("\n[Phase 1] Expert Warm-Start (A* trajectories)...")
    observer = Observer()
    navigator = Navigator()
    warmup_hist = warm_start(observer, navigator)
    
    # Phase 2
    print("\n[Phase 2] RL Fine-Tuning...")
    rl_hist = rl_fine_tune(observer, navigator)
    
    # Evaluate
    print("\n" + "=" * 40)
    print("EVALUATION (200 mazes):")
    eval_results = evaluate(observer, navigator)
    print(f"  Neuralese: {eval_results['neur_success']:.1%} success, {eval_results['neur_steps']:.1f} avg steps, {eval_results['wall_hits_neur']:.1f} wall hits/ep")
    print(f"  A* (optimal): {eval_results['greedy_success']:.1%} success, {eval_results['greedy_steps']:.1f} avg steps")
    
    # Latent evolution
    print("\n[Latent Evolution Test]")
    latent_evolution_test(observer, navigator, out_dir)
    
    # Visualize
    print("\nGenerating plots...")
    plot_summary(warmup_hist, rl_hist, eval_results, out_dir)
    
    print(f"\nDone! Outputs in {out_dir}/")
