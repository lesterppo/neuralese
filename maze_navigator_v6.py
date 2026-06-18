"""
Neuralese Remote Brain v6 — Linear Diversity Decay + Wall Proximity Features
Key improvements over v5:
1. Wall-proximity features in Observer input (explicit distances to nearest walls)
2. Linear diversity decay (not multiplicative — stays higher longer)
3. Longer training (8000 episodes) with per-step wall proximity penalty
4. Continuous wall-avoidance reward (not collision-only)
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
BATCH = 16
WARMUP_EPOCHS = 1000
PPO_EPOCHS = 4
PPO_EPISODES = 8000     # Longer run
PPO_CLIP = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.99
LR = 1e-3
RL_LR = 1e-3
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
DIVERSITY_WEIGHT_INIT = 0.15    # Linear decay from 0.15→0.01
DIVERSITY_WEIGHT_FINAL = 0.01
WALL_PROXIMITY_PENALTY = -0.1   # Per-step penalty scaled by inverse distance to nearest wall
WALL_COLLISION_PENALTY = -10.0
GOAL_REWARD = 10.0
STEP_PENALTY = -0.05
TARGET_THRESH = 0.5

# Wall proximity features: 5 extra input dims for Observer
# [dist_up, dist_down, dist_left, dist_right, wall_count_3x3]
WALL_FEATURE_DIM = 5

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
        if (r, c) == target: return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc)); q.append((nr, nc))
    return False

def get_radar(grid, pos_r, pos_c):
    radar = torch.ones(3, 3)
    for i in range(3):
        for j in range(3):
            nr, nc = int(pos_r) + i - 1, int(pos_c) + j - 1
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                radar[i, j] = grid[nr, nc]
    return radar.flatten()

def get_wall_proximity(grid, pos_r, pos_c):
    """Compute distance to nearest wall in 4 directions + wall count in 3×3.
    Returns a 5D feature vector normalized to [0, 1]."""
    r, c = int(pos_r), int(pos_c)
    r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))

    # Distance to nearest wall in each direction (capped at GRID_SIZE)
    dist_up = 0
    for dr in range(1, GRID_SIZE):
        nr = r - dr
        if nr < 0 or grid[nr, c] == 1.0:
            dist_up = dr; break
    if dist_up == 0: dist_up = GRID_SIZE

    dist_down = 0
    for dr in range(1, GRID_SIZE):
        nr = r + dr
        if nr >= GRID_SIZE or grid[nr, c] == 1.0:
            dist_down = dr; break
    if dist_down == 0: dist_down = GRID_SIZE

    dist_left = 0
    for dc in range(1, GRID_SIZE):
        nc = c - dc
        if nc < 0 or grid[r, nc] == 1.0:
            dist_left = dc; break
    if dist_left == 0: dist_left = GRID_SIZE

    dist_right = 0
    for dc in range(1, GRID_SIZE):
        nc = c + dc
        if nc >= GRID_SIZE or grid[r, nc] == 1.0:
            dist_right = dc; break
    if dist_right == 0: dist_right = GRID_SIZE

    # Wall count in 3×3 neighborhood
    wall_count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if grid[nr, nc] == 1.0:
                    wall_count += 1

    return torch.tensor([
        dist_up / GRID_SIZE, dist_down / GRID_SIZE,
        dist_left / GRID_SIZE, dist_right / GRID_SIZE,
        wall_count / 8.0,  # Max 8 neighbors (excluding center)
    ], dtype=torch.float32)


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
                path.append(curr); curr = parent[curr]
            path.reverse(); return path
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


class ObserverActorCritic(nn.Module):
    """MLP Observer with wall-proximity features: grid(100) + pos(2) + wall_feats(5) → 12D Neuralese + V."""
    input_dim = GRID_SIZE * GRID_SIZE + 2 + WALL_FEATURE_DIM  # 107
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(HIDDEN, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )
        self.critic_head = nn.Linear(HIDDEN, 1)

    def forward(self, grid_flat, pos, wall_feats):
        pos_norm = pos / GRID_SIZE
        x = torch.cat([grid_flat, pos_norm, wall_feats], dim=-1)
        shared_feat = self.shared(x)
        z = self.actor_head(shared_feat)
        v = self.critic_head(shared_feat.detach())
        return z, v

    def get_latent(self, grid_flat, pos, wall_feats):
        pos_norm = pos / GRID_SIZE
        x = torch.cat([grid_flat, pos_norm, wall_feats], dim=-1)
        shared_feat = self.shared(x)
        return self.actor_head(shared_feat)


class Navigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(9 + LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu_head = nn.Sequential(nn.Linear(HIDDEN, 2), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, radar, z, deterministic=False):
        feat = self.shared(torch.cat([radar, z], dim=-1))
        mu = self.mu_head(feat) * STEP_CAP
        std = torch.exp(self.log_std).expand_as(mu)
        if deterministic: return mu
        return D.Normal(mu, std)


# --- WARM-START ---

def generate_expert_batch(batch_size, wall_prob=WALL_PROB):
    all_items = []
    for _ in range(batch_size):
        grid, start, target = gen_maze(wall_prob)
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 2:
            for i in range(len(path) - 1):
                curr_r, curr_c = path[i]
                next_r, next_c = path[i+1]
                move = torch.tensor([next_r - curr_r, next_c - curr_c], dtype=torch.float32)
                pos = torch.tensor([curr_r, curr_c], dtype=torch.float32)
                radar = get_radar(grid, curr_r, curr_c)
                wall_f = get_wall_proximity(grid, curr_r, curr_c)
                all_items.append((grid.flatten(), pos, wall_f, radar, move))
    return all_items

def warm_start(observer, navigator, epochs=WARMUP_EPOCHS):
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []
    for ep in range(epochs):
        expert_items = generate_expert_batch(BATCH)
        if len(expert_items) < BATCH: continue
        idx = np.random.choice(len(expert_items), BATCH, replace=False)
        grid_batch = torch.stack([expert_items[i][0] for i in idx])
        pos_batch = torch.stack([expert_items[i][1] for i in idx])
        wall_batch = torch.stack([expert_items[i][2] for i in idx])
        radar_batch = torch.stack([expert_items[i][3] for i in idx])
        target_move = torch.stack([expert_items[i][4] for i in idx])
        z = observer.get_latent(grid_batch, pos_batch, wall_batch)
        pred_move = navigator(radar_batch, z, deterministic=True)
        loss = nn.MSELoss()(pred_move, target_move) + 0.001 * torch.norm(z, p=2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step(); scheduler.step()
        history.append(loss.item())
        if ep % 500 == 0: print(f"  Warm-up {ep:5d}: loss={loss.item():.6f}")
    return history


# --- PPO with Linear Diversity Decay ---

def compute_wall_proximity_reward(grids, positions):
    """Per-step continuous wall proximity penalty: rewards being far from walls."""
    B = len(grids)
    rewards = torch.zeros(B)
    for b in range(B):
        r, c = int(round(positions[b, 0].item())), int(round(positions[b, 1].item()))
        r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))
        # Min distance to any wall in 4 cardinal directions
        min_dist = GRID_SIZE
        for dr in range(1, GRID_SIZE):
            nr = r + dr
            if nr >= GRID_SIZE or grids[b][nr, c] == 1.0:
                min_dist = min(min_dist, dr); break
        for dr in range(1, GRID_SIZE):
            nr = r - dr
            if nr < 0 or grids[b][nr, c] == 1.0:
                min_dist = min(min_dist, dr); break
        for dc in range(1, GRID_SIZE):
            nc = c + dc
            if nc >= GRID_SIZE or grids[b][r, nc] == 1.0:
                min_dist = min(min_dist, dc); break
        for dc in range(1, GRID_SIZE):
            nc = c - dc
            if nc < 0 or grids[b][r, nc] == 1.0:
                min_dist = min(min_dist, dc); break
        # Reward: log distance (encourages staying away, but diminishing returns)
        if min_dist > 0:
            rewards[b] = np.log(1 + min_dist) * 0.1  # Small positive for being far
        else:
            rewards[b] = WALL_PROXIMITY_PENALTY  # Adjacent to wall
    return rewards


def collect_trajectories(observer, navigator, num_eps):
    all_obs_grids, all_obs_pos, all_obs_walls = [], [], []
    all_radars, all_actions, all_log_probs = [], [], []
    all_rewards, all_values, all_terminated, all_zs = [], [], [], []

    for _ in range(num_eps):
        grid, start, target = gen_maze()
        grid_flat = grid.flatten()
        pos = start.clone()
        active = True
        ep_grids, ep_pos, ep_walls = [], [], []
        ep_radars, ep_actions, ep_log_probs = [], [], []
        ep_rewards, ep_values, ep_terminated, ep_zs = [], [], [], []

        for _ in range(MAX_STEPS):
            if not active: break
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            wall_f = get_wall_proximity(grid, r, c)

            with torch.no_grad():
                z, v = observer(grid_flat.unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
                dist = navigator(radar.unsqueeze(0), z)
                action = dist.sample().squeeze(0)
                log_prob = dist.log_prob(action).sum(dim=-1)

            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r_new, c_new = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r_new, c_new = max(0, min(GRID_SIZE-1, r_new)), max(0, min(GRID_SIZE-1, c_new))

            wall_hit = (grid[r_new, c_new] == 1.0)
            reached = torch.norm(new_pos - target).item() < TARGET_THRESH
            reward = -torch.norm(new_pos - target).item() * 0.1 + STEP_PENALTY
            is_terminal = False
            if wall_hit:
                reward += WALL_COLLISION_PENALTY
                active = False; is_terminal = True
            if reached:
                reward += GOAL_REWARD
                active = False; is_terminal = True

            ep_grids.append(grid_flat)
            ep_pos.append(pos)
            ep_walls.append(wall_f)
            ep_radars.append(radar)
            ep_actions.append(action)
            ep_log_probs.append(log_prob.item())
            ep_rewards.append(reward)
            ep_values.append(v.item())
            ep_terminated.append(is_terminal)
            ep_zs.append(z.squeeze(0))
            pos = new_pos

        if ep_grids:
            all_obs_grids.append(torch.stack(ep_grids))
            all_obs_pos.append(torch.stack(ep_pos))
            all_obs_walls.append(torch.stack(ep_walls))
            all_radars.append(torch.stack(ep_radars))
            all_actions.append(torch.stack(ep_actions))
            all_log_probs.append(ep_log_probs)
            all_rewards.append(ep_rewards)
            all_values.append(ep_values)
            all_terminated.append(ep_terminated)
            all_zs.append(ep_zs)

    return (all_obs_grids, all_obs_pos, all_obs_walls, all_radars, all_actions,
            all_log_probs, all_rewards, all_values, all_terminated, all_zs)


def compute_gae(rewards_list, values_list, terminated_list, observer, all_obs_grids, all_obs_pos, all_obs_walls):
    all_advantages, all_returns = [], []
    for i in range(len(rewards_list)):
        T = len(rewards_list[i])
        advantages = torch.zeros(T); returns = torch.zeros(T)

        last_step_v = 0.0
        if not terminated_list[i][-1]:
            with torch.no_grad():
                _, v_last = observer(all_obs_grids[i][-1:], all_obs_pos[i][-1:], all_obs_walls[i][-1:])
                last_step_v = v_last.item()

        gae = 0.0; next_value = last_step_v
        for t in reversed(range(T)):
            mask = 0.0 if terminated_list[i][t] else 1.0
            delta = rewards_list[i][t] + GAMMA * next_value * mask - values_list[i][t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[t] = gae; returns[t] = gae + values_list[i][t]
            next_value = 0.0 if terminated_list[i][t] else values_list[i][t]
        all_advantages.append(advantages); all_returns.append(returns)
    return all_advantages, all_returns


def ppo_update(observer, navigator, opt, trajectories, advantages_list, returns_list,
               all_zs_rollout, div_weight):
    (all_obs_grids, all_obs_pos, all_obs_walls, all_radars, all_actions,
     all_log_probs, _, all_values, _, _) = trajectories

    grids_cat = torch.cat([g for g in all_obs_grids], dim=0)
    pos_cat = torch.cat([p for p in all_obs_pos], dim=0)
    walls_cat = torch.cat([w for w in all_obs_walls], dim=0)
    radars_cat = torch.cat([r for r in all_radars], dim=0)
    actions_cat = torch.cat([a for a in all_actions], dim=0)
    old_log_probs_cat = torch.tensor([lp for lps in all_log_probs for lp in lps], dtype=torch.float32)
    advantages_cat = torch.cat([a for a in advantages_list], dim=0)
    returns_cat = torch.cat([r for r in returns_list], dim=0)
    advantages_cat = (advantages_cat - advantages_cat.mean()) / (advantages_cat.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        z_current, values = observer(grids_cat, pos_cat, walls_cat)
        dist = navigator(radars_cat, z_current)
        new_log_probs = dist.log_prob(actions_cat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_cat)
        surr1 = ratio * advantages_cat
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages_cat
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values.squeeze(-1), returns_cat)

        # Diversity loss on CURRENT policy latents (per-episode drift)
        total_drift = 0.0; n_pairs = 0; offset = 0
        for ep_zs_rollout in all_zs_rollout:
            ep_len = len(ep_zs_rollout)
            if ep_len < 2: offset += ep_len; continue
            z_ep = z_current[offset:offset + ep_len]
            for t in range(1, ep_len):
                total_drift += torch.norm(z_ep[t] - z_ep[t-1], p=2); n_pairs += 1
            offset += ep_len
        avg_drift = total_drift / max(1, n_pairs)
        diversity_loss = -div_weight * avg_drift

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy + diversity_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 0.5)
        opt.step()

    return policy_loss.item(), value_loss.item(), entropy.item(), diversity_loss.item()


# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=200):
    neur_success, neur_steps, wall_hits = 0, 0, 0
    for _ in range(num_eps):
        grid, start, target = gen_maze()
        pos = start.clone()
        for step in range(MAX_STEPS):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            wall_f = get_wall_proximity(grid, r, c)
            with torch.no_grad():
                z, _ = observer(grid.flatten().unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
                action = navigator(radar.unsqueeze(0), z, deterministic=True).squeeze(0)
            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r_new, c_new = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r_new, c_new = max(0, min(GRID_SIZE-1, r_new)), max(0, min(GRID_SIZE-1, c_new))
            if grid[r_new, c_new] == 1.0: wall_hits += 1; break
            pos = new_pos
            if torch.norm(pos - target).item() < TARGET_THRESH:
                neur_success += 1; neur_steps += step + 1; break
        else: neur_steps += MAX_STEPS

    greedy_success, greedy_steps = 0, 0
    for _ in range(num_eps):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path: greedy_success += 1; greedy_steps += len(path) - 1
        else: greedy_steps += MAX_STEPS

    n_eval = max(1, neur_success if neur_success > 0 else num_eps)
    return {"neur_success": neur_success / num_eps, "neur_steps": neur_steps / n_eval,
            "greedy_success": greedy_success / num_eps, "greedy_steps": greedy_steps / max(1, greedy_success),
            "wall_hits_neur": wall_hits / num_eps}


def latent_evolution_test(observer, navigator, out_dir):
    for attempt in range(20):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 5 and np.sum(grid.numpy()) > 2: break
    print(f"  Path length: {len(path)}, Walls: {int(grid.sum().item())}")
    pos = start.clone(); latents, positions = [], []
    for step in range(min(len(path), 30)):
        radar = get_radar(grid, int(pos[0].item()), int(pos[1].item()))
        wall_f = get_wall_proximity(grid, int(pos[0].item()), int(pos[1].item()))
        with torch.no_grad():
            z, _ = observer(grid.flatten().unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
            action = navigator(radar.unsqueeze(0), z, deterministic=True).squeeze(0)
        latents.append(z.squeeze(0).numpy()); positions.append(pos.clone().numpy())
        new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
        r, c = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
        if grid[r, c] == 1.0: break
        pos = new_pos
        if torch.norm(pos - target).item() < TARGET_THRESH: break

    latents_arr = np.array(latents); positions_arr = np.array(positions)
    z_std = np.std(latents_arr, axis=0).mean()
    distances = [np.linalg.norm(latents_arr[i] - latents_arr[0]) for i in range(len(latents_arr))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]; ax.plot(distances, 'b-o', markersize=8)
    ax.set_title("Latent Distance from t=0"); ax.set_xlabel("Step"); ax.set_ylabel("||z_t - z_0||"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    for d in range(min(8, LATENT_DIM)): ax.plot(latents_arr[:, d], alpha=0.5, label=f"dim {d}", marker='o', markersize=4)
    ax.set_title("Latent Dimensions over Steps"); ax.set_xlabel("Step"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax = axes[2]; ax.imshow(grid.numpy().T, cmap='gray_r', origin='lower', alpha=0.3)
    ax.plot(positions_arr[:, 0], positions_arr[:, 1], 'b-', linewidth=2, label='Agent')
    ax.scatter(positions_arr[0, 0], positions_arr[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(target[0].item(), target[1].item(), c='red', s=150, marker='*', label='Target')
    if path:
        path_arr = np.array(path); ax.plot(path_arr[:, 0], path_arr[:, 1], 'r--', alpha=0.3, label='A* path')
    ax.set_xlim(-0.5, GRID_SIZE-0.5); ax.set_ylim(-0.5, GRID_SIZE-0.5); ax.set_title("Trajectory"); ax.legend(fontsize=7)
    plt.suptitle(f"Latent Evolution — z_std={z_std:.4f} ({'EMERGENT' if z_std > 0.05 else 'STATIC'})", fontsize=14)
    plt.tight_layout(); plt.savefig(out_dir / "maze_latent_evolution_v6.png", dpi=150); plt.close()
    print(f"  Latent std: {z_std:.4f} → {'EMERGENT' if z_std > 0.05 else 'STATIC'}")
    print(f"  Max drift: {max(distances):.4f}")
    return z_std, latents_arr


def plot_summary(warmup_hist, ppo_history, eval_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]; ax.plot(warmup_hist, alpha=0.5, color='blue')
    ax.set_title("Warm-Start"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]; ax.plot(ppo_history["success"], alpha=0.6, color='orange', label="Success")
    ax.plot(ppo_history["wall_hits"], alpha=0.6, color='red', label="Walls/ep")
    ax.set_title("PPO Training"); ax.set_xlabel("Eval #"); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    x = np.arange(2); width = 0.35
    ax.bar(x - width/2, [eval_results["neur_success"], eval_results["greedy_success"]], width, label="Success Rate", color="blue", alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, [eval_results["neur_steps"], eval_results["greedy_steps"]], width, label="Avg Steps", color="red", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Neuralese", "A*"]); ax.set_ylabel("Success Rate"); ax2.set_ylabel("Avg Steps")
    ax.set_title("Final Evaluation"); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
    ax = axes[1, 1]
    ax.text(0.1, 0.5,
            f"Linear div decay: {DIVERSITY_WEIGHT_INIT}→{DIVERSITY_WEIGHT_FINAL}\n"
            f"Wall proximity features: {WALL_FEATURE_DIM}D\n"
            f"Wall hits/ep: {eval_results['wall_hits_neur']:.2f}\n"
            f"Episodes: {PPO_EPISODES}",
            fontfamily='monospace', fontsize=10, va='center', transform=ax.transAxes)
    ax.set_title("Config"); ax.axis('off')
    plt.suptitle("Neuralese v6 — Linear Decay + Wall Features", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir / "maze_results_v6.png", dpi=150); plt.close()


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("NEURALESE v6 — Linear Diversity Decay + Wall Features")
    print("=" * 60)
    print(f"  Wall features: {WALL_FEATURE_DIM}D (4-direction distances + 3×3 count)")
    print(f"  Diversity: {DIVERSITY_WEIGHT_INIT}→{DIVERSITY_WEIGHT_FINAL} linear decay")
    print(f"  Wall proximity reward: continuous (log distance)")

    print("\n[Phase 1] Warm-Start...")
    observer = ObserverActorCritic()
    navigator = Navigator()
    warmup_hist = warm_start(observer, navigator)

    print("\n[Phase 2] PPO + Linear Diversity Decay...")
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    ppo_history = {"success": [], "wall_hits": [], "entropy": [], "div_loss": []}
    ep_counter = 0; episodes_per_update = 32

    while ep_counter < PPO_EPISODES:
        trajectories = collect_trajectories(observer, navigator, episodes_per_update)
        all_obs_grids, all_obs_pos, all_obs_walls, all_radars, all_actions, \
            all_log_probs, all_rewards, all_values, all_terminated, all_zs = trajectories

        advantages, returns = compute_gae(all_rewards, all_values, all_terminated,
                                          observer, all_obs_grids, all_obs_pos, all_obs_walls)

        # Linear diversity decay: interpolate from init to final
        progress = min(1.0, ep_counter / PPO_EPISODES)
        div_weight = DIVERSITY_WEIGHT_INIT + (DIVERSITY_WEIGHT_FINAL - DIVERSITY_WEIGHT_INIT) * progress

        p_loss, v_loss, ent, div = ppo_update(observer, navigator, opt, trajectories,
                                               advantages, returns, all_zs, div_weight)
        ep_counter += episodes_per_update

        if ep_counter % 200 == 0 or ep_counter >= PPO_EPISODES:
            eval_r = evaluate(observer, navigator, num_eps=50)
            ppo_history["success"].append(eval_r["neur_success"])
            ppo_history["wall_hits"].append(eval_r["wall_hits_neur"])
            ppo_history["entropy"].append(ent)
            ppo_history["div_loss"].append(div)
            print(f"  PPO ep {ep_counter:5d}: succ={eval_r['neur_success']:.1%} "
                  f"walls={eval_r['wall_hits_neur']:.2f} ent={ent:.3f} "
                  f"div={div:.4f} div_w={div_weight:.3f}")

    print("\n" + "=" * 40)
    print("FINAL EVALUATION (200 mazes):")
    eval_results = evaluate(observer, navigator, num_eps=200)
    print(f"  Neuralese: {eval_results['neur_success']:.1%}, {eval_results['neur_steps']:.1f} steps, "
          f"{eval_results['wall_hits_neur']:.2f} walls/ep")
    print(f"  A*: {eval_results['greedy_success']:.1%}, {eval_results['greedy_steps']:.1f} steps")

    print("\n[Latent Evolution Test]")
    latent_evolution_test(observer, navigator, out_dir)
    print("\nGenerating plots...")
    plot_summary(warmup_hist, ppo_history, eval_results, out_dir)
    print(f"\nDone! Outputs in {out_dir}/")
