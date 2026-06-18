"""
Neuralese Remote Brain v7 — Temporal Stacking + Auxiliary Reconstruction
Key fixes from Gemini Flash review (Session 4, c_7d0b925d438def85):

1. STRIP continuous wall proximity reward → use delta-Manhattan-distance shaping
   (Proximity reward was TEACHING the Navigator to loiter in safe open areas)
2. STACK previous N=3 Neuralese vectors [z_t, z_{t-1}, z_{t-2}] for temporal context
   (Single vector lacks history → Navigator suffers representational amnesia)
3. AUXILIARY RECONSTRUCTION: Navigator predicts wall distances from Neuralese
   (Forces Navigator to actually LISTEN — if it can't predict walls, communication is broken)

Architecture:
  Observer: grid(100) + pos(2) + wall_feats(5) = 107D → 12D Neuralese + V(s)
  Navigator: radar(9) + z_t(12) + z_{t-1}(12) + z_{t-2}(12) = 45D → movement(2) + wall_pred(5)
  Loss: PPO + 0.5*Value + 0.005*Entropy - div_weight*Diversity + 0.1*WallReconstruction
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
PPO_EPISODES = 6000
PPO_CLIP = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.99
LR = 1e-3
RL_LR = 1e-3
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
DIVERSITY_WEIGHT = 0.10        # Fixed moderate diversity (adaptive bumps if stalled)
WALL_RECON_COEF = 0.1           # Auxiliary reconstruction loss weight
WALL_COLLISION_PENALTY = -10.0
GOAL_REWARD = 10.0
STEP_PENALTY = -0.05
TARGET_THRESH = 0.5
WALL_FEATURE_DIM = 5           # [dist_up, dist_down, dist_left, dist_right, wall_count_3x3]
TEMPORAL_WINDOW = 3             # Stack current + 2 previous z vectors

# --- ENVIRONMENT ---

def gen_maze(wall_prob=WALL_PROB):
    for _ in range(100):
        grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < wall_prob).astype(np.float32)
        start, target = (0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)
        grid[start] = 0.0; grid[target] = 0.0
        if bfs_reachable(grid, start, target):
            return torch.tensor(grid), torch.tensor(start, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    return torch.tensor(grid), torch.tensor((0, 0), dtype=torch.float32), torch.tensor((GRID_SIZE-1, GRID_SIZE-1), dtype=torch.float32)

def bfs_reachable(grid, start, target):
    q = deque([start]); visited = {start}
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
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE: radar[i, j] = grid[nr, nc]
    return radar.flatten()

def get_wall_proximity(grid, pos_r, pos_c):
    r, c = int(pos_r), int(pos_c)
    r, c = max(0, min(GRID_SIZE-1, r)), max(0, min(GRID_SIZE-1, c))
    dist_up = GRID_SIZE
    for dr in range(1, GRID_SIZE):
        nr = r - dr
        if nr < 0 or grid[nr, c] == 1.0: dist_up = dr; break
    dist_down = GRID_SIZE
    for dr in range(1, GRID_SIZE):
        nr = r + dr
        if nr >= GRID_SIZE or grid[nr, c] == 1.0: dist_down = dr; break
    dist_left = GRID_SIZE
    for dc in range(1, GRID_SIZE):
        nc = c - dc
        if nc < 0 or grid[r, nc] == 1.0: dist_left = dc; break
    dist_right = GRID_SIZE
    for dc in range(1, GRID_SIZE):
        nc = c + dc
        if nc >= GRID_SIZE or grid[r, nc] == 1.0: dist_right = dc; break
    wall_count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 1.0: wall_count += 1
    return torch.tensor([dist_up/GRID_SIZE, dist_down/GRID_SIZE, dist_left/GRID_SIZE,
                         dist_right/GRID_SIZE, wall_count/8.0], dtype=torch.float32)

def a_star_path(grid, start, target):
    sr, sc = int(start[0].item()), int(start[1].item())
    tr, tc = int(target[0].item()), int(target[1].item())
    g_score = {(sr, sc): 0}; parent = {(sr, sc): None}
    open_set = [(abs(sr-tr) + abs(sc-tc), sr, sc)]
    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r, c) == (tr, tc):
            path = []; curr = (tr, tc)
            while curr is not None: path.append(curr); curr = parent[curr]
            path.reverse(); return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                tentative_g = g_score[(r, c)] + 1
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative_g; parent[(nr, nc)] = (r, c)
                    f = tentative_g + abs(nr-tr) + abs(nc-tc)
                    heapq.heappush(open_set, (f, nr, nc))
    return None


# --- MODELS ---

class ObserverActorCritic(nn.Module):
    input_dim = GRID_SIZE * GRID_SIZE + 2 + WALL_FEATURE_DIM  # 107
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.actor_head = nn.Sequential(nn.Linear(HIDDEN, LATENT_DIM), nn.LayerNorm(LATENT_DIM))
        self.critic_head = nn.Linear(HIDDEN, 1)

    def forward(self, grid_flat, pos, wall_feats):
        x = torch.cat([grid_flat, pos / GRID_SIZE, wall_feats], dim=-1)
        shared_feat = self.shared(x)
        return self.actor_head(shared_feat), self.critic_head(shared_feat.detach())

    def get_latent(self, grid_flat, pos, wall_feats):
        x = torch.cat([grid_flat, pos / GRID_SIZE, wall_feats], dim=-1)
        return self.actor_head(self.shared(x))


class Navigator(nn.Module):
    """Input: radar(9) + z_t(12) + z_{t-1}(12) + z_{t-2}(12) = 45D
       Output: movement mean(2) + wall_prediction(5)"""
    input_dim = 9 + LATENT_DIM * TEMPORAL_WINDOW  # 9 + 36 = 45
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu_head = nn.Sequential(nn.Linear(HIDDEN, 2), nn.Tanh())
        self.wall_pred_head = nn.Linear(HIDDEN, WALL_FEATURE_DIM)  # Predict wall distances
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, radar, z_stacked, deterministic=False):
        """z_stacked: [B, LATENT_DIM * TEMPORAL_WINDOW]"""
        feat = self.shared(torch.cat([radar, z_stacked], dim=-1))
        mu = self.mu_head(feat) * STEP_CAP
        wall_pred = self.wall_pred_head(feat)
        std = torch.exp(self.log_std).expand_as(mu)
        if deterministic: return mu, wall_pred
        return D.Normal(mu, std), wall_pred


# --- WARM-START ---

def generate_expert_batch(batch_size, wall_prob=WALL_PROB):
    all_items = []
    for _ in range(batch_size):
        grid, start, target = gen_maze(wall_prob)
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 2:
            for i in range(len(path) - 1):
                curr_r, curr_c = path[i]; next_r, next_c = path[i+1]
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
        # For warm-start: zero-pad temporal context
        z_stacked = torch.cat([z, torch.zeros_like(z), torch.zeros_like(z)], dim=-1)
        pred_move, wall_pred = navigator(radar_batch, z_stacked, deterministic=True)
        # Task loss + wall reconstruction
        task_loss = nn.MSELoss()(pred_move, target_move)
        wall_loss = nn.MSELoss()(wall_pred, wall_batch)
        loss = task_loss + 0.1 * wall_loss + 0.001 * torch.norm(z, p=2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 1.0)
        opt.step(); scheduler.step()
        history.append(loss.item())
        if ep % 500 == 0: print(f"  Warm-up {ep:5d}: loss={loss.item():.6f} task={task_loss.item():.4f} wall={wall_loss.item():.4f}")
    return history


# --- PPO with Adaptive Diversity ---

def collect_trajectories(observer, navigator, num_eps):
    """Rollouts with temporal stacking. Tracks previous 2 z vectors for each agent."""
    all_obs_grids, all_obs_pos, all_obs_walls = [], [], []
    all_radars, all_z_stacked, all_actions = [], [], []
    all_log_probs, all_rewards, all_values, all_terminated = [], [], [], []
    all_zs, all_prev_dists = [], []  # For diversity loss and Manhattan shaping

    for _ in range(num_eps):
        grid, start, target = gen_maze()
        grid_flat = grid.flatten(); pos = start.clone(); active = True
        ep_grids, ep_pos, ep_walls, ep_radars = [], [], [], []
        ep_z_stacked, ep_actions, ep_log_probs = [], [], []
        ep_rewards, ep_values, ep_terminated, ep_zs = [], [], [], []
        z_history = []  # Sliding window of last 2 z vectors
        prev_dist = torch.norm(pos - target).item()  # For Manhattan shaping

        for _ in range(MAX_STEPS):
            if not active: break
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c); wall_f = get_wall_proximity(grid, r, c)

            with torch.no_grad():
                z, v = observer(grid_flat.unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
                # Build temporal stack: [z_t, z_{t-1}, z_{t-2}] zero-padded
                z_prev = z_history[-2:] if len(z_history) >= 2 else \
                         ([torch.zeros_like(z)] * (2 - len(z_history)) + z_history)
                z_st = torch.cat([z] + z_prev, dim=-1)  # [1, 36]
                dist, wall_pred = navigator(radar.unsqueeze(0), z_st)
                action = dist.sample().squeeze(0)
                log_prob = dist.log_prob(action).sum(dim=-1)

            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r_new, c_new = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r_new, c_new = max(0, min(GRID_SIZE-1, r_new)), max(0, min(GRID_SIZE-1, c_new))

            wall_hit = (grid[r_new, c_new] == 1.0)
            reached = torch.norm(new_pos - target).item() < TARGET_THRESH
            new_dist = torch.norm(new_pos - target).item()

            # Manhattan distance shaping: reward reduction in distance to goal
            reward = (prev_dist - new_dist) * 1.0 + STEP_PENALTY  # Positive for getting closer
            is_terminal = False
            if wall_hit:
                reward += WALL_COLLISION_PENALTY
                active = False; is_terminal = True
            if reached:
                reward += GOAL_REWARD
                active = False; is_terminal = True

            ep_grids.append(grid_flat); ep_pos.append(pos); ep_walls.append(wall_f)
            ep_radars.append(radar); ep_z_stacked.append(z_st.squeeze(0))
            ep_actions.append(action); ep_log_probs.append(log_prob.item())
            ep_rewards.append(reward); ep_values.append(v.item())
            ep_terminated.append(is_terminal); ep_zs.append(z.squeeze(0))

            z_history.append(z); prev_dist = new_dist; pos = new_pos
            if len(z_history) > 2: z_history.pop(0)

        if ep_grids:
            all_obs_grids.append(torch.stack(ep_grids)); all_obs_pos.append(torch.stack(ep_pos))
            all_obs_walls.append(torch.stack(ep_walls)); all_radars.append(torch.stack(ep_radars))
            all_z_stacked.append(torch.stack(ep_z_stacked)); all_actions.append(torch.stack(ep_actions))
            all_log_probs.append(ep_log_probs); all_rewards.append(ep_rewards)
            all_values.append(ep_values); all_terminated.append(ep_terminated)
            all_zs.append(ep_zs); all_prev_dists.append(prev_dist)

    return (all_obs_grids, all_obs_pos, all_obs_walls, all_radars, all_z_stacked,
            all_actions, all_log_probs, all_rewards, all_values, all_terminated, all_zs)


def compute_gae(rewards_list, values_list, terminated_list, observer, all_obs_grids, all_obs_pos, all_obs_walls):
    all_advantages, all_returns = [], []
    for i in range(len(rewards_list)):
        T = len(rewards_list[i]); advantages = torch.zeros(T); returns = torch.zeros(T)
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
               all_zs_rollout, all_obs_walls, div_weight):
    (all_obs_grids, all_obs_pos, all_obs_walls_traj, all_radars, all_z_stacked,
     all_actions, all_log_probs, _, all_values, _, _) = trajectories

    grids_cat = torch.cat([g for g in all_obs_grids], dim=0)
    pos_cat = torch.cat([p for p in all_obs_pos], dim=0)
    walls_cat_in = torch.cat([w for w in all_obs_walls_traj], dim=0)
    radars_cat = torch.cat([r for r in all_radars], dim=0)
    z_stacked_cat = torch.cat([zs for zs in all_z_stacked], dim=0)
    actions_cat = torch.cat([a for a in all_actions], dim=0)
    old_log_probs_cat = torch.tensor([lp for lps in all_log_probs for lp in lps], dtype=torch.float32)
    advantages_cat = torch.cat([a for a in advantages_list], dim=0)
    returns_cat = torch.cat([r for r in returns_list], dim=0)
    advantages_cat = (advantages_cat - advantages_cat.mean()) / (advantages_cat.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        z_current, values = observer(grids_cat, pos_cat, walls_cat_in)
        # Rebuild temporal stack from current policy's z (NOT rollout cache!)
        z_stacked_current = build_temporal_stack(z_current, all_zs_rollout, grids_cat.shape[0])
        dist, wall_pred = navigator(radars_cat, z_stacked_current)
        new_log_probs = dist.log_prob(actions_cat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_cat)
        surr1 = ratio * advantages_cat
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages_cat
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values.squeeze(-1), returns_cat)

        # Auxiliary wall reconstruction loss
        wall_loss = nn.MSELoss()(wall_pred, walls_cat_in)

        # Diversity loss on CURRENT policy latents
        total_drift = 0.0; n_pairs = 0; offset = 0
        for ep_zs_r in all_zs_rollout:
            ep_len = len(ep_zs_r)
            if ep_len < 2: offset += ep_len; continue
            z_ep = z_current[offset:offset + ep_len]
            for t in range(1, ep_len):
                total_drift += torch.norm(z_ep[t] - z_ep[t-1], p=2); n_pairs += 1
            offset += ep_len
        diversity_loss = -div_weight * (total_drift / max(1, n_pairs))

        loss = (policy_loss + VALUE_COEF * value_loss + WALL_RECON_COEF * wall_loss
                - ENTROPY_COEF * entropy + diversity_loss)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(navigator.parameters()), 0.5)
        opt.step()

    return policy_loss.item(), value_loss.item(), entropy.item(), diversity_loss.item(), wall_loss.item()


def build_temporal_stack(z_current, all_zs_rollout, total_N):
    """Rebuild temporal context [z_t, z_{t-1}, z_{t-2}] from current policy z values.
    Uses rollout z history for previous steps to avoid sequential recomputation."""
    z_stacked = torch.zeros(total_N, LATENT_DIM * TEMPORAL_WINDOW, device=z_current.device)
    offset = 0
    for ep_zs_r in all_zs_rollout:
        ep_len = len(ep_zs_r)
        for t in range(ep_len):
            idx = offset + t
            # Current z (from current policy)
            stack = [z_current[idx]]
            # Previous z (from rollout history, which is a reasonable approximation)
            for k in range(1, TEMPORAL_WINDOW):
                if t - k >= 0:
                    prev_z = z_current[offset + t - k]  # Use current policy's z for history too
                else:
                    prev_z = torch.zeros(LATENT_DIM, device=z_current.device)
                stack.append(prev_z)
            z_stacked[idx] = torch.cat(stack, dim=-1)
        offset += ep_len
    return z_stacked


# --- EVALUATION ---

def evaluate(observer, navigator, num_eps=200):
    neur_success, neur_steps, wall_hits = 0, 0, 0
    for _ in range(num_eps):
        grid, start, target = gen_maze(); pos = start.clone()
        z_history = []
        for step in range(MAX_STEPS):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c); wall_f = get_wall_proximity(grid, r, c)
            with torch.no_grad():
                z, _ = observer(grid.flatten().unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
                z_prev = z_history[-2:] if len(z_history) >= 2 else (
                    [torch.zeros_like(z)] * (2 - len(z_history)) + z_history)
                z_st = torch.cat([z] + z_prev, dim=-1)
                action, _ = navigator(radar.unsqueeze(0), z_st, deterministic=True)
                action = action.squeeze(0)
            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            r_new, c_new = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            r_new, c_new = max(0, min(GRID_SIZE-1, r_new)), max(0, min(GRID_SIZE-1, c_new))
            if grid[r_new, c_new] == 1.0: wall_hits += 1; break
            pos = new_pos; z_history.append(z)
            if len(z_history) > 2: z_history.pop(0)
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
    return {"neur_success": neur_success/num_eps, "neur_steps": neur_steps/n_eval,
            "greedy_success": greedy_success/num_eps, "greedy_steps": greedy_steps/max(1, greedy_success),
            "wall_hits_neur": wall_hits/num_eps}


def latent_evolution_test(observer, navigator, out_dir):
    for attempt in range(20):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path) >= 5 and np.sum(grid.numpy()) > 2: break
    print(f"  Path: {len(path)}, Walls: {int(grid.sum().item())}")
    pos = start.clone(); latents, positions, z_history = [], [], []
    for step in range(min(len(path), 30)):
        radar = get_radar(grid, int(pos[0].item()), int(pos[1].item()))
        wall_f = get_wall_proximity(grid, int(pos[0].item()), int(pos[1].item()))
        with torch.no_grad():
            z, _ = observer(grid.flatten().unsqueeze(0), pos.unsqueeze(0), wall_f.unsqueeze(0))
            z_prev = z_history[-2:] if len(z_history) >= 2 else (
                [torch.zeros_like(z)] * (2 - len(z_history)) + z_history)
            z_st = torch.cat([z] + z_prev, dim=-1)
            action, _ = navigator(radar.unsqueeze(0), z_st, deterministic=True)
            action = action.squeeze(0)
        latents.append(z.squeeze(0).numpy()); positions.append(pos.clone().numpy())
        new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
        r, c = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
        if grid[r, c] == 1.0: break
        pos = new_pos; z_history.append(z)
        if len(z_history) > 2: z_history.pop(0)
        if torch.norm(pos - target).item() < TARGET_THRESH: break

    latents_arr = np.array(latents); z_std = np.std(latents_arr, axis=0).mean()
    distances = [np.linalg.norm(latents_arr[i] - latents_arr[0]) for i in range(len(latents_arr))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(distances, 'b-o', markersize=8); axes[0].set_title("Latent Distance from t=0"); axes[0].grid(True, alpha=0.3)
    for d in range(min(8, LATENT_DIM)): axes[1].plot(latents_arr[:, d], alpha=0.5, label=f"dim {d}", marker='o', markersize=4)
    axes[1].set_title("Latent Dimensions"); axes[1].legend(fontsize=6); axes[1].grid(True, alpha=0.3)
    axes[2].imshow(grid.numpy().T, cmap='gray_r', origin='lower', alpha=0.3)
    axes[2].plot(np.array(positions)[:, 0], np.array(positions)[:, 1], 'b-', linewidth=2)
    axes[2].scatter(start[0].item(), start[1].item(), c='green', s=100, marker='o')
    axes[2].scatter(target[0].item(), target[1].item(), c='red', s=150, marker='*')
    axes[2].set_xlim(-0.5, GRID_SIZE-0.5); axes[2].set_ylim(-0.5, GRID_SIZE-0.5); axes[2].set_title("Trajectory")
    plt.suptitle(f"Latent Evolution v7 — std={z_std:.4f} ({'EMERGENT' if z_std>0.05 else 'STATIC'})", fontsize=14)
    plt.tight_layout(); plt.savefig(out_dir/"maze_latent_evolution_v7.png", dpi=150); plt.close()
    print(f"  z_std: {z_std:.4f} → {'EMERGENT' if z_std>0.05 else 'STATIC'}, max drift: {max(distances):.4f}")
    return z_std, latents_arr


def plot_summary(warmup_hist, ppo_history, eval_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(warmup_hist, alpha=0.5, color='blue'); axes[0, 0].set_title("Warm-Start"); axes[0, 0].set_yscale("log"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(ppo_history["success"], alpha=0.6, color='orange', label="Success")
    axes[0, 1].plot(ppo_history["wall_hits"], alpha=0.6, color='red', label="Walls/ep")
    axes[0, 1].set_title("PPO Training"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    x = np.arange(2); width = 0.35
    axes[1, 0].bar(x-width/2, [eval_results["neur_success"], eval_results["greedy_success"]], width, label="Success", color="blue", alpha=0.7)
    ax2 = axes[1, 0].twinx()
    ax2.bar(x+width/2, [eval_results["neur_steps"], eval_results["greedy_steps"]], width, label="Steps", color="red", alpha=0.7)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(["Neuralese", "A*"]); axes[1, 0].set_ylim(0, 1.1); axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 1].text(0.1, 0.5,
                    f"v7: Temporal stacking (N=3)\nAux wall reconstruction (α={WALL_RECON_COEF})\n"
                    f"Manhattan distance shaping\nFixed div={DIVERSITY_WEIGHT}\n"
                    f"Wall hits: {eval_results['wall_hits_neur']:.2f}",
                    fontfamily='monospace', fontsize=10, va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Config"); axes[1, 1].axis('off')
    plt.suptitle("Neuralese v7 — Temporal Context + Auxiliary Reconstruction", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/"maze_results_v7.png", dpi=150); plt.close()


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("NEURALESE v7 — Temporal Stacking + Auxiliary Reconstruction")
    print("=" * 60)
    print(f"  Temporal window: {TEMPORAL_WINDOW} z-vectors stacked")
    print(f"  Rewards: delta-Manhattan (NOT continuous proximity)")
    print(f"  Aux loss: wall recon (α={WALL_RECON_COEF})")
    print(f"  Navigator input: {Navigator.input_dim}D")

    print("\n[Phase 1] Warm-Start...")
    observer = ObserverActorCritic(); navigator = Navigator()
    warmup_hist = warm_start(observer, navigator)

    print("\n[Phase 2] PPO Training (adaptive diversity)...")
    opt = optim.Adam(list(observer.parameters()) + list(navigator.parameters()), lr=RL_LR)
    ppo_history = {"success": [], "wall_hits": [], "entropy": [], "div_loss": [], "wall_loss": []}
    ep_counter = 0; episodes_per_update = 32
    div_weight = DIVERSITY_WEIGHT
    last_success = 0.0; stall_count = 0

    while ep_counter < PPO_EPISODES:
        trajectories = collect_trajectories(observer, navigator, episodes_per_update)
        (all_obs_grids, all_obs_pos, all_obs_walls, all_radars, all_z_stacked,
         all_actions, all_log_probs, all_rewards, all_values, all_terminated, all_zs) = trajectories

        advantages, returns = compute_gae(all_rewards, all_values, all_terminated,
                                          observer, all_obs_grids, all_obs_pos, all_obs_walls)

        p_loss, v_loss, ent, div, wall_l = ppo_update(observer, navigator, opt, trajectories,
                                                       advantages, returns, all_zs, all_obs_walls,
                                                       div_weight)
        ep_counter += episodes_per_update

        if ep_counter % 200 == 0 or ep_counter >= PPO_EPISODES:
            eval_r = evaluate(observer, navigator, num_eps=50)
            ppo_history["success"].append(eval_r["neur_success"])
            ppo_history["wall_hits"].append(eval_r["wall_hits_neur"])
            ppo_history["entropy"].append(ent)
            ppo_history["div_loss"].append(div)
            ppo_history["wall_loss"].append(wall_l)

            # Adaptive diversity: if success stalls for 5 evals, bump diversity
            if eval_r["neur_success"] <= last_success + 0.02:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= 5:
                div_weight = min(0.3, div_weight * 1.5)
                stall_count = 0
                print(f"    → Adaptive bump: div_weight = {div_weight:.3f}")
            last_success = eval_r["neur_success"]

            print(f"  PPO ep {ep_counter:5d}: succ={eval_r['neur_success']:.1%} "
                  f"walls={eval_r['wall_hits_neur']:.2f} ent={ent:.3f} "
                  f"div={div:.4f} wall_l={wall_l:.4f} div_w={div_weight:.3f}")

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
