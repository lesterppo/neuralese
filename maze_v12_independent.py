"""
Neuralese v12 — Independent Agents + GRU Baseline + Fixed Reward + Info Plane
=============================================================================
THE DEFINITIVE EXPERIMENT. Tests whether emergent communication actually happens.

KEY CHANGES FROM v5/v11:
  1. INDEPENDENT AGENTS: Observer and Navigator trained with SEPARATE optimizers
     and separate value functions. Observer's ONLY job is to communicate usefully.
  2. GRU BASELINE: Recurrent Navigator with NO Observer. If GRU >= Neuralese,
     the Observer→z pathway is unnecessary. This is the falsification test.
  3. FIXED REWARDS: Continuous distance shaping (not -10 wall bomb).
     Wall adjacency penalty instead of collision-only.
  4. INFORMATION PLANE: I(X;Z) vs I(Z;Y) analysis to quantify compression quality.

ARCHITECTURE:
  Observer (sees full state)          Navigator (local radar only)
  grid(100)+pos(2)+target(2)=104D      radar(9)+z(8)=17D
         │                                    │
         ▼                                    ▼
      Observer MLP                        Navigator MLP
         │                                    │
         ▼                                    ▼
     8D Neuralese z ────────────────────► PPO action distribution
         │                                    │
         ▼                                    ▼
   V_obs(s) critic                      V_nav(radar,z) critic

  GRU Baseline: radar(9)+pos(2)+target(2)=13D → GRU → action
  No Observer. No channel. Just memory.

NULL-CHANNEL: After training, test Navigator with random z. If performance
  doesn't drop, the channel carries no information regardless of training.

SUCCESS CRITERIA:
  Neuralese > GRU: Channel is useful (hypothesis supported)
  Neuralese ≈ GRU: Channel adds nothing (hypothesis falsified)
  Neuralese >> Null: Channel encodes real information (emergence)
  Neuralese ≈ Null: Observer learned nothing useful
"""

import torch, torch.nn as nn, torch.optim as optim, torch.distributions as D
import numpy as np
from pathlib import Path; from collections import deque
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- CONFIG ---
GRID_SIZE = 10; WALL_PROB = 0.20; LATENT_DIM = 8; HIDDEN = 128
MAX_STEPS = 50; STEP_CAP = 0.5
BATCH_EPISODES = 32; PPO_EPOCHS = 3; TOTAL_EPISODES = 6000
PPO_CLIP = 0.2; GAE_LAMBDA = 0.95; GAMMA = 0.99
LR = 3e-4; ENTROPY_COEF = 0.01; VALUE_COEF = 0.5
TARGET_THRESH = 0.5

# --- FIXED REWARDS: continuous shaping, no -10 wall bomb ---
GOAL_REWARD = 5.0            # Substantial but not overwhelming
DISTANCE_WEIGHT = 0.2        # Reward for reducing distance to goal
WALL_ADJACENCY_PENALTY = -0.1  # Small penalty per wall in radar (nudges away)
STEP_PENALTY = -0.01         # Tiny — don't punish exploration
WALL_COLLISION_PENALTY = -2.0  # Collision penalty (not -10, allows recovery learning)
TIMEOUT_PENALTY = -1.0        # Didn't reach goal in MAX_STEPS

# --- ENVIRONMENT ---

def gen_maze():
    for _ in range(200):
        g = (np.random.rand(GRID_SIZE, GRID_SIZE) < WALL_PROB).astype(np.float32)
        s, t = (0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)
        g[s] = 0.0; g[t] = 0.0
        if bfs_reachable(g, s, t):
            return (torch.tensor(g),
                    torch.tensor(s, dtype=torch.float32),
                    torch.tensor(t, dtype=torch.float32))
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    return torch.tensor(g), torch.tensor((0,0),dtype=torch.float32), torch.tensor((9,9),dtype=torch.float32)

def bfs_reachable(g, s, t):
    q = deque([s]); v = {s}
    while q:
        r, c = q.popleft()
        if (r, c) == t: return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and g[nr,nc]==0 and (nr,nc) not in v:
                v.add((nr,nc)); q.append((nr,nc))
    return False

def get_radar(grid, r, c):
    radar = torch.ones(3, 3)
    for i in range(3):
        for j in range(3):
            nr, nc = r+i-1, c+j-1
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE:
                radar[i,j] = grid[nr,nc]
    return radar.flatten()

# --- AGENT ARCHITECTURES ---

class Observer(nn.Module):
    """Sees full state → outputs 8D Neuralese + value estimate.
    Trained INDEPENDENTLY from Navigator. Reward = Navigator's environmental reward."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(GRID_SIZE*GRID_SIZE + 4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.z_head = nn.Sequential(nn.Linear(HIDDEN, LATENT_DIM), nn.LayerNorm(LATENT_DIM))
        self.v_head = nn.Linear(HIDDEN, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, grid_flat, pos, target):
        x = torch.cat([grid_flat, pos/GRID_SIZE, target/GRID_SIZE], dim=-1)
        h = self.shared(x)
        z = self.z_head(h)
        v = self.v_head(h)
        return z, v


class Navigator(nn.Module):
    """Local radar + Observer's z → action distribution + value.
    Trained INDEPENDENTLY from Observer."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(9 + LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(HIDDEN, 2), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2))
        self.v_head = nn.Linear(HIDDEN, 1)

    def forward(self, radar, z, deterministic=False):
        h = self.shared(torch.cat([radar, z], dim=-1))
        h = torch.clamp(h, -10, 10)
        mu = self.mu_head(h) * STEP_CAP
        mu = torch.clamp(mu, -STEP_CAP, STEP_CAP)
        v = self.v_head(h)
        if deterministic:
            return mu, v
        return D.Normal(mu, torch.exp(self.log_std).expand_as(mu) + 1e-6), v


class GRUNavigator(nn.Module):
    """Recurrent Navigator: radar + position + target → GRU → action.
    NO Observer. NO channel. This is the falsification baseline."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(9 + 2 + 2, HIDDEN)  # radar + pos + target
        self.mu_head = nn.Sequential(nn.Linear(HIDDEN, 2), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2))
        self.v_head = nn.Linear(HIDDEN, 1)
        # Xavier init to prevent NaN on first forward
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, radar, pos, target, hidden=None, deterministic=False):
        if hidden is None:
            hidden = torch.zeros(radar.size(0), HIDDEN, device=radar.device)
        x = torch.cat([radar, pos/GRID_SIZE, target/GRID_SIZE], dim=-1)
        h = self.gru(x, hidden)
        h = torch.clamp(h, -10, 10)  # Prevent exploding activations
        mu = self.mu_head(h) * STEP_CAP
        mu = torch.clamp(mu, -STEP_CAP, STEP_CAP)  # Ensure mu stays in range
        v = self.v_head(h)
        if deterministic:
            return mu, h, v
        return D.Normal(mu, torch.exp(self.log_std).expand_as(mu) + 1e-6), h, v


# --- ROLLOUT COLLECTION ---

def collect_episodes_neur(obs, nav, n_eps):
    """Collect episodes for Neuralese Observer+Navigator pair."""
    data = {
        "obs_grids": [], "obs_pos": [], "obs_targets": [],
        "nav_radars": [], "nav_zs": [],
        "actions": [], "logps": [], "rewards": [],
        "obs_values": [], "nav_values": [], "terms": []
    }
    for _ in range(n_eps):
        grid, start, target = gen_maze()
        gflat = grid.flatten(); pos = start.clone(); active = True
        ep = {k: [] for k in data}
        for st in range(MAX_STEPS):
            if not active: break
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                z, v_obs = obs(gflat.unsqueeze(0), pos.unsqueeze(0), target.unsqueeze(0))
                dist, v_nav = nav(radar.unsqueeze(0), z)
                action = dist.sample().squeeze(0)
                logp = dist.log_prob(action).sum(dim=-1)

            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            rn, cn = max(0, min(GRID_SIZE-1, rn)), max(0, min(GRID_SIZE-1, cn))

            prev_dist = torch.norm(pos - target).item()
            new_dist = torch.norm(new_pos - target).item()
            wall_hit = (grid[rn, cn] == 1.0)
            reached = new_dist < TARGET_THRESH

            # Fixed reward: continuous shaping
            reward = (prev_dist - new_dist) * DISTANCE_WEIGHT + STEP_PENALTY
            # Wall adjacency penalty
            walls_in_radar = radar.sum().item()
            reward += walls_in_radar * WALL_ADJACENCY_PENALTY
            term = False

            if wall_hit:
                reward += WALL_COLLISION_PENALTY
                active = False; term = True
            if reached:
                reward += GOAL_REWARD
                active = False; term = True
            if st == MAX_STEPS - 1 and not reached:
                reward += TIMEOUT_PENALTY
                term = True

            ep["obs_grids"].append(gflat); ep["obs_pos"].append(pos)
            ep["obs_targets"].append(target); ep["nav_radars"].append(radar)
            ep["nav_zs"].append(z.squeeze(0)); ep["actions"].append(action)
            ep["logps"].append(logp.item()); ep["rewards"].append(reward)
            ep["obs_values"].append(v_obs.item()); ep["nav_values"].append(v_nav.item())
            ep["terms"].append(term)
            pos = new_pos

        if ep["obs_grids"]:
            for k in data: data[k].append(ep[k])
    return data


def collect_episodes_gru(gru, n_eps):
    """Collect episodes for GRU baseline (no Observer)."""
    data = {
        "radars": [], "pos": [], "targets": [],
        "actions": [], "logps": [], "rewards": [],
        "values": [], "terms": []
    }
    for _ in range(n_eps):
        grid, start, target = gen_maze()
        pos = start.clone(); active = True; hidden = None
        ep = {k: [] for k in data}
        for st in range(MAX_STEPS):
            if not active: break
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                dist, hidden, v = gru(radar.unsqueeze(0), pos.unsqueeze(0),
                                      target.unsqueeze(0), hidden)
                action = dist.sample().squeeze(0)
                logp = dist.log_prob(action).sum(dim=-1)

            new_pos = torch.clamp(pos + action, 0, GRID_SIZE - 1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            rn, cn = max(0, min(GRID_SIZE-1, rn)), max(0, min(GRID_SIZE-1, cn))

            prev_dist = torch.norm(pos - target).item()
            new_dist = torch.norm(new_pos - target).item()
            wall_hit = (grid[rn, cn] == 1.0)
            reached = new_dist < TARGET_THRESH

            reward = (prev_dist - new_dist) * DISTANCE_WEIGHT + STEP_PENALTY
            reward += radar.sum().item() * WALL_ADJACENCY_PENALTY
            term = False

            if wall_hit:
                reward += WALL_COLLISION_PENALTY
                active = False; term = True
            if reached:
                reward += GOAL_REWARD
                active = False; term = True
            if st == MAX_STEPS - 1 and not reached:
                reward += TIMEOUT_PENALTY
                term = True

            ep["radars"].append(radar); ep["pos"].append(pos)
            ep["targets"].append(target); ep["actions"].append(action)
            ep["logps"].append(logp.item()); ep["rewards"].append(reward)
            ep["values"].append(v.item()); ep["terms"].append(term)
            pos = new_pos

        if ep["radars"]:
            for k in data: data[k].append(ep[k])
    return data


# --- GAE ---

def compute_gae(rewards, values, terms):
    advs, rets = [], []
    for i in range(len(rewards)):
        rw, val, term = rewards[i], values[i], terms[i]
        T = len(rw)
        a = torch.zeros(T); ret = torch.zeros(T)
        gae = 0.0; next_v = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if term[t] else 1.0
            delta = rw[t] + GAMMA * next_v * mask - val[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            a[t] = gae; ret[t] = gae + val[t]
            next_v = 0.0 if term[t] else val[t]
        advs.append(a); rets.append(ret)
    return advs, rets


# --- INDEPENDENT PPO UPDATES ---

def ppo_update_obs(obs, opt_obs, data, advs, rets):
    """Update Observer using its OWN value function. PPO clipped objective."""
    grids = torch.stack([g for ep in data["obs_grids"] for g in ep])
    pos = torch.stack([p for ep in data["obs_pos"] for p in ep])
    targets = torch.stack([t for ep in data["obs_targets"] for t in ep])
    old_lps = torch.tensor([lp for ep_lps in data["logps"] for lp in ep_lps], dtype=torch.float32)
    adv_cat = torch.cat(advs); ret_cat = torch.cat(rets)
    adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)

    # Also need log probs from Navigator's action distribution
    radars = torch.stack([r for ep in data["nav_radars"] for r in ep])
    nav = data.get("_nav_ref")  # We need the Navigator reference

    for _ in range(PPO_EPOCHS):
        z, obs_vals = obs(grids, pos, targets)
        # Get Navigator's distribution using THIS z
        if nav is not None:
            dist, nav_vals = nav(radars, z)
            new_lps = dist.log_prob(torch.stack([a for ep in data["actions"] for a in ep])).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            new_lps = old_lps  # fallback
            entropy = 0.0

        ratio = torch.exp(new_lps - old_lps)
        s1 = ratio * adv_cat; s2 = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP) * adv_cat
        policy_loss = -torch.min(s1, s2).mean()
        value_loss = nn.MSELoss()(obs_vals.squeeze(-1), ret_cat)

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        opt_obs.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(obs.parameters(), 0.5)
        opt_obs.step()

    return policy_loss.item(), value_loss.item()


def ppo_update_nav(nav, opt_nav, data, advs, rets):
    """Update Navigator using its OWN value function."""
    radars = torch.stack([r for ep in data["nav_radars"] for r in ep])
    zs = torch.stack([z for ep in data["nav_zs"] for z in ep])
    acts = torch.stack([a for ep in data["actions"] for a in ep])
    old_lps = torch.tensor([lp for ep_lps in data["logps"] for lp in ep_lps], dtype=torch.float32)
    adv_cat = torch.cat(advs); ret_cat = torch.cat(rets)
    adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        dist, nav_vals = nav(radars, zs)
        new_lps = dist.log_prob(acts).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_lps - old_lps)
        s1 = ratio * adv_cat; s2 = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP) * adv_cat
        policy_loss = -torch.min(s1, s2).mean()
        value_loss = nn.MSELoss()(nav_vals.squeeze(-1), ret_cat)

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        opt_nav.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(nav.parameters(), 0.5)
        opt_nav.step()

    return policy_loss.item(), value_loss.item()


def ppo_update_gru(gru, opt, data, advs, rets):
    """Update GRU Navigator."""
    acts = torch.stack([a for ep in data["actions"] for a in ep])
    old_lps = torch.tensor([lp for ep_lps in data["logps"] for lp in ep_lps], dtype=torch.float32)
    adv_cat = torch.cat(advs); ret_cat = torch.cat(rets)
    adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)

    total_loss = torch.tensor(0.0)
    total_count = 0; count = 0
    for i in range(len(data["radars"])):
        r = data["radars"][i]; p = data["pos"][i]; t = data["targets"][i]
        a = data["actions"][i]; ep_adv = advs[i]; ep_ret = rets[i]
        hidden = None; ep_lps = []; ep_vals = []

        # Re-forward to get current policy's log-probs and values
        for j in range(len(r)):
            dist, hidden, v = gru(r[j].unsqueeze(0), p[j].unsqueeze(0),
                                  t[j].unsqueeze(0), hidden)
            ep_lps.append(dist.log_prob(a[j].unsqueeze(0)).sum(dim=-1))
            ep_vals.append(v)

        nlps = torch.stack(ep_lps).squeeze()
        vals = torch.stack(ep_vals).squeeze()

        if len(r) < 2:
            count += len(r); continue
        ep_adv_n = (ep_adv - ep_adv.mean()) / (ep_adv.std() + 1e-8)
        if torch.isnan(ep_adv_n).any():
            count += len(r); continue

        ratio = torch.exp(nlps - old_lps[count:count+len(r)])
        s1 = ratio * ep_adv_n; s2 = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP) * ep_adv_n
        p_loss = -torch.min(s1, s2).mean()
        v_loss = nn.MSELoss()(vals, ep_ret)
        total_loss = total_loss + p_loss + VALUE_COEF * v_loss
        total_count += 1
        count += len(r)

    if total_count > 0:
        opt.zero_grad()
        (total_loss / total_count).backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), 0.5)
        opt.step()
    return total_loss.item()


# --- EVALUATION ---

def evaluate_all(obs, nav, gru, n_eps=200):
    """Evaluate Neuralese, Null Channel, and GRU."""
    # Neuralese
    ns, nw = 0, 0
    for _ in range(n_eps):
        grid, start, target = gen_maze(); pos = start.clone()
        for st in range(MAX_STEPS):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                z, _ = obs(grid.flatten().unsqueeze(0), pos.unsqueeze(0), target.unsqueeze(0))
                a, _ = nav(radar.unsqueeze(0), z, deterministic=True)
                a = a.squeeze(0)
            new_pos = torch.clamp(pos + a, 0, GRID_SIZE-1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            rn, cn = max(0, min(GRID_SIZE-1, rn)), max(0, min(GRID_SIZE-1, cn))
            if grid[rn, cn] == 1.0: nw += 1; break
            pos = new_pos
            if torch.norm(pos - target).item() < TARGET_THRESH: ns += 1; break

    # Null Channel (replace z with random noise)
    null_s, null_w = 0, 0
    for _ in range(n_eps):
        grid, start, target = gen_maze(); pos = start.clone()
        for st in range(MAX_STEPS):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                z_noise = torch.randn(1, LATENT_DIM) * 0.5
                a, _ = nav(radar.unsqueeze(0), z_noise, deterministic=True)
                a = a.squeeze(0)
            new_pos = torch.clamp(pos + a, 0, GRID_SIZE-1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            rn, cn = max(0, min(GRID_SIZE-1, rn)), max(0, min(GRID_SIZE-1, cn))
            if grid[rn, cn] == 1.0: null_w += 1; break
            pos = new_pos
            if torch.norm(pos - target).item() < TARGET_THRESH: null_s += 1; break

    # GRU baseline
    gs, gw = 0, 0
    for _ in range(n_eps):
        grid, start, target = gen_maze(); pos = start.clone(); hidden = None
        for st in range(MAX_STEPS):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                a, hidden, _ = gru(radar.unsqueeze(0), pos.unsqueeze(0),
                                   target.unsqueeze(0), hidden, deterministic=True)
                a = a.squeeze(0)
            new_pos = torch.clamp(pos + a, 0, GRID_SIZE-1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            rn, cn = max(0, min(GRID_SIZE-1, rn)), max(0, min(GRID_SIZE-1, cn))
            if grid[rn, cn] == 1.0: gw += 1; break
            pos = new_pos
            if torch.norm(pos - target).item() < TARGET_THRESH: gs += 1; break

    return {
        "neur_succ": ns/n_eps, "neur_walls": nw/n_eps,
        "null_succ": null_s/n_eps, "null_walls": null_w/n_eps,
        "gru_succ": gs/n_eps, "gru_walls": gw/n_eps,
    }


# --- INFORMATION PLANE ---

def info_plane(obs, nav, n_samples=500):
    """Estimate I(X;Z) and I(Z;Y) for the communication channel.
    X = Observer's input (full state), Z = Neuralese vector, Y = Navigator's output.
    Uses kernel-density mutual information estimation (simple binning approach)."""
    xs, zs, ys = [], [], []
    for _ in range(n_samples):
        grid, start, target = gen_maze()
        pos = start.clone()
        for st in range(min(20, MAX_STEPS)):
            r, c = int(pos[0].item()), int(pos[1].item())
            radar = get_radar(grid, r, c)
            with torch.no_grad():
                z, _ = obs(grid.flatten().unsqueeze(0), pos.unsqueeze(0), target.unsqueeze(0))
                a, _ = nav(radar.unsqueeze(0), z, deterministic=True)
            xs.append(pos.numpy().copy())  # Observer input summary: position
            zs.append(z.squeeze(0).numpy())
            ys.append(a.squeeze(0).numpy())
            new_pos = torch.clamp(pos + a.squeeze(0), 0, GRID_SIZE-1)
            rn, cn = int(round(new_pos[0].item())), int(round(new_pos[1].item()))
            if grid[rn, cn] == 1.0: break
            pos = new_pos
            if torch.norm(pos - target).item() < TARGET_THRESH: break

    xs = np.array(xs); zs = np.array(zs); ys = np.array(ys)

    # Per-dimension correlation as proxy for MI
    z_var = np.var(zs, axis=0).mean()
    z_std = np.std(zs, axis=0).mean()

    # X→Z: variance captured
    # Z→Y: how much does z predict action?
    y_corrs = []
    for d in range(LATENT_DIM):
        for yd in range(2):  # action x, y
            c = np.corrcoef(zs[:, d], ys[:, yd])[0, 1]
            y_corrs.append(abs(c))

    max_action_corr = max(y_corrs)
    mean_action_corr = np.mean(y_corrs)

    # Active dimensions: how many z dims have any correlation with output
    active_dims = sum(1 for d in range(LATENT_DIM)
                      if max(abs(np.corrcoef(zs[:, d], ys[:, 0])[0, 1]),
                             abs(np.corrcoef(zs[:, d], ys[:, 1])[0, 1])) > 0.1)

    return {
        "z_variance": z_var, "z_std_mean": z_std,
        "max_action_corr": max_action_corr, "mean_action_corr": mean_action_corr,
        "active_dims": active_dims, "total_dims": LATENT_DIM,
        "zs": zs, "ys": ys, "xs": xs,
    }


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEURALESE v12 — Independent Agents + GRU Baseline + Info Plane")
    print("=" * 70)
    print(f"  Bottleneck: {LATENT_DIM}D")
    print(f"  Observer: full state (104D) → {LATENT_DIM}D z")
    print(f"  Navigator: radar (9D) + z ({LATENT_DIM}D) → action")
    print(f"  GRU: radar + pos + target → GRU → action (NO Observer)")
    print(f"  Reward: continuous distance shaping (not -10 wall bomb)")
    print(f"  Training: {TOTAL_EPISODES} episodes, {BATCH_EPISODES}/batch")
    print()

    # Initialize agents with SEPARATE optimizers
    obs = Observer()
    nav = Navigator()
    gru = GRUNavigator()

    opt_obs = optim.Adam(obs.parameters(), lr=LR)
    opt_nav = optim.Adam(nav.parameters(), lr=LR)
    opt_gru = optim.Adam(gru.parameters(), lr=LR)

    hist = {
        "neur_succ": [], "null_succ": [], "gru_succ": [],
        "neur_walls": [], "gru_walls": [],
        "z_active_dims": [], "z_max_corr": [],
    }

    ep_count = 0
    print("[Training] Independent agents + GRU baseline...")
    while ep_count < TOTAL_EPISODES:
        # Collect rollouts
        data_neur = collect_episodes_neur(obs, nav, BATCH_EPISODES)
        data_gru = collect_episodes_gru(gru, BATCH_EPISODES)

        # Compute GAE for each value function separately
        advs_obs, rets_obs = compute_gae(data_neur["rewards"], data_neur["obs_values"], data_neur["terms"])
        advs_nav, rets_nav = compute_gae(data_neur["rewards"], data_neur["nav_values"], data_neur["terms"])
        advs_gru, rets_gru = compute_gae(data_gru["rewards"], data_gru["values"], data_gru["terms"])

        # Store Navigator reference for Observer update
        data_neur["_nav_ref"] = nav
        ppo_update_obs(obs, opt_obs, data_neur, advs_obs, rets_obs)
        ppo_update_nav(nav, opt_nav, data_neur, advs_nav, rets_nav)
        ppo_update_gru(gru, opt_gru, data_gru, advs_gru, rets_gru)

        ep_count += BATCH_EPISODES

        # Evaluate periodically
        if ep_count % 400 == 0 or ep_count >= TOTAL_EPISODES:
            ev = evaluate_all(obs, nav, gru, n_eps=50)

            # Info plane snapshot
            ip = info_plane(obs, nav, n_samples=100)

            hist["neur_succ"].append(ev["neur_succ"])
            hist["null_succ"].append(ev["null_succ"])
            hist["gru_succ"].append(ev["gru_succ"])
            hist["neur_walls"].append(ev["neur_walls"])
            hist["gru_walls"].append(ev["gru_walls"])
            hist["z_active_dims"].append(ip["active_dims"])
            hist["z_max_corr"].append(ip["max_action_corr"])

            print(f"  ep{ep_count:5d}: neur={ev['neur_succ']:.1%} null={ev['null_succ']:.1%} "
                  f"gru={ev['gru_succ']:.1%} | z_act={ip['active_dims']}/{LATENT_DIM} "
                  f"z_corr={ip['max_action_corr']:.3f}")

    # --- FINAL EVALUATION ---
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (200 mazes each)")
    ev = evaluate_all(obs, nav, gru, n_eps=200)
    ip = info_plane(obs, nav, n_samples=500)

    print(f"  Neuralese:   {ev['neur_succ']:.1%} success, {ev['neur_walls']:.2f} walls/ep")
    print(f"  Null Channel: {ev['null_succ']:.1%} success, {ev['null_walls']:.2f} walls/ep")
    print(f"  GRU (no Obs): {ev['gru_succ']:.1%} success, {ev['gru_walls']:.2f} walls/ep")
    print(f"  Channel info: {ev['neur_succ'] - ev['null_succ']:+.1%} over null")
    print(f"  Observer value: {ev['neur_succ'] - ev['gru_succ']:+.1%} over GRU")
    print(f"  Active z dims: {ip['active_dims']}/{LATENT_DIM}")
    print(f"  Max z→action corr: {ip['max_action_corr']:.3f}")
    print(f"  Mean z→action corr: {ip['mean_action_corr']:.3f}")

    # --- VERDICT ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    neur_over_null = ev["neur_succ"] - ev["null_succ"]
    neur_over_gru = ev["neur_succ"] - ev["gru_succ"]

    if neur_over_null < 0.05:
        print("  >> NULL HYPOTHESIS NOT REJECTED")
        print("  >> The Observer→z channel carries no significant information.")
        print("  >> Navigator relies on its own radar, not the Observer's instructions.")
    elif neur_over_gru <= 0:
        print("  >> GRU BASELINE MATCHES OR BEATS NEURALESE")
        print("  >> The Observer adds no value over a simple recurrent policy.")
        print("  >> Neuralese hypothesis FALSIFIED for this architecture.")
    elif ip["active_dims"] < 2:
        print("  >> INSUFFICIENT CHANNEL UTILIZATION")
        print(f"  >> Only {ip['active_dims']}/{LATENT_DIM} latent dimensions carry information.")
        print("  >> The bottleneck is mostly unused — not genuine communication.")
    else:
        print("  >> CHANNEL SHOWS EMERGENT COMMUNICATION")
        print(f"  >> Neuralese beats null by {neur_over_null:+.1%} and GRU by {neur_over_gru:+.1%}")
        print(f"  >> {ip['active_dims']}/{LATENT_DIM} dimensions actively encode task information.")
        print("  >> Neuralese hypothesis SUPPORTED for this architecture.")

    # --- PLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(hist["neur_succ"], 'b-o', label='Neuralese')
    ax.plot(hist["null_succ"], 'r--s', label='Null Channel')
    ax.plot(hist["gru_succ"], 'orange', marker='^', label='GRU (no Observer)')
    ax.set_title("Success Rate"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[0, 1]
    width = 0.25; x = np.arange(3)
    bars = ax.bar(x, [ev["neur_succ"], ev["null_succ"], ev["gru_succ"]], width,
                  color=['blue', 'red', 'orange'], alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Neuralese", "Null Channel", "GRU"])
    ax.set_title("Final Comparison")
    for bar, v in zip(bars, [ev["neur_succ"], ev["null_succ"], ev["gru_succ"]]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.1%}", ha='center')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1)

    ax = axes[0, 2]
    ax.plot(hist["z_active_dims"], 'purple', marker='o', label='Active z dims')
    ax.plot(hist["z_max_corr"], 'green', marker='s', label='Max z→action corr')
    ax.set_title("Channel Utilization"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(LATENT_DIM, 1.1))

    ax = axes[1, 0]
    ax.plot(hist["neur_walls"], 'b-o', label='Neuralese walls')
    ax.plot(hist["gru_walls"], 'orange', marker='^', label='GRU walls')
    ax.set_title("Wall Collisions per Episode"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if ip["zs"] is not None and ip["ys"] is not None:
        zs = ip["zs"]; ys = ip["ys"]
        corr_matrix = np.zeros((LATENT_DIM, 2))
        for d in range(LATENT_DIM):
            corr_matrix[d, 0] = np.corrcoef(zs[:, d], ys[:, 0])[0, 1]
            corr_matrix[d, 1] = np.corrcoef(zs[:, d], ys[:, 1])[0, 1]
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Action X", "Action Y"])
        ax.set_ylabel("Latent dim"); ax.set_title("z → Action Correlation")
        plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    verdict_text = (
        f"FINAL RESULTS\n"
        f"─────────────\n"
        f"Neuralese: {ev['neur_succ']:.1%}\n"
        f"Null Channel: {ev['null_succ']:.1%}\n"
        f"GRU baseline: {ev['gru_succ']:.1%}\n"
        f"\nChannel value:\n"
        f"  vs Null: {neur_over_null:+.1%}\n"
        f"  vs GRU: {neur_over_gru:+.1%}\n"
        f"\nInfo plane:\n"
        f"  Active: {ip['active_dims']}/{LATENT_DIM} dims\n"
        f"  Max corr: {ip['max_action_corr']:.3f}"
    )
    ax.text(0.1, 0.5, verdict_text, fontfamily='monospace', fontsize=9,
            va='center', transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle("Neuralese v12 — Independent Agents + GRU Baseline + Info Plane",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "maze_v12_results.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir}/maze_v12_results.png")

    # Save info plane data
    info_data = {
        "z_variance": float(ip["z_variance"]),
        "z_std_mean": float(ip["z_std_mean"]),
        "max_action_corr": float(ip["max_action_corr"]),
        "mean_action_corr": float(ip["mean_action_corr"]),
        "active_dims": int(ip["active_dims"]),
        "neur_succ": float(ev["neur_succ"]),
        "null_succ": float(ev["null_succ"]),
        "gru_succ": float(ev["gru_succ"]),
        "neur_over_null": float(neur_over_null),
        "neur_over_gru": float(neur_over_gru),
    }
    import json
    with open(out_dir / "maze_v12_info.json", "w") as f:
        json.dump(info_data, f, indent=2)
    print(f"Saved: {out_dir}/maze_v12_info.json")
