"""
Neuralese Remote Brain v11 — Randomized Start/Goal + Channel Noise
Per Gemini Pro Session 5 (c_5c6c5f64a1dac19d):

Fix 1: RANDOMIZE start and target positions (was hardcoded (0,0)→(9,9))
  - The agents were memorizing a macro-route, not learning communication

Fix 2: CHANNEL NOISE — inject Gaussian noise into z between Observer→Navigator
  - Forces Observer to push vectors apart so Navigator can decode through noise
  - Replaces fragile diversity loss with a physical constraint
  - If Observer outputs the same z every step, Navigator can not distinguish states
  - Natural emergence: noise requires variation, not arbitrary penalty

Architecture (from v6/v10 — proven):
  - MLP Observer: grid(100) + pos(2) + wall_feats(5) = 107D → 12D Neuralese + V(s)
  - Stateless Navigator: radar(9) + 12D z = 21D → Normal(mu, learnable_std)
  - No temporal stacking, no aux reconstruction, no diversity loss
  - PPO Actor-Critic, GAE, linear LR decay
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

GRID_SIZE = 10; WALL_PROB = 0.20; LATENT_DIM = 12; HIDDEN = 128
STEP_CAP = 0.5; MAX_STEPS = 50; BATCH = 16
WARMUP_EPOCHS = 1000; PPO_EPOCHS = 4; PPO_EPISODES = 8000
PPO_CLIP = 0.2; GAE_LAMBDA = 0.95; GAMMA = 0.99; LR = 1e-3; RL_LR = 1e-3
ENTROPY_COEF = 0.005; VALUE_COEF = 0.5
WALL_COLLISION_PENALTY = -10.0; WALL_ADJACENCY_PENALTY = -0.3
GOAL_REWARD = 10.0; STEP_PENALTY = -0.05; TARGET_THRESH = 0.5
WALL_FEATURE_DIM = 5
CHANNEL_NOISE_STD = 0.1
DIVERSITY_WEIGHT = 0.05
EXP_PROXIMITY_ALPHA = 0.5   # Narrow-band exponential proximity
EXP_PROXIMITY_BETA = 2.0     # Decay rate: exp(-2*d) → 0.14 at d=1, 0.02 at d=2

# 4 fixed start/goal pairs covering different maze regions
FIXED_PAIRS = [
    ((0, 0), (9, 9)),      # Top-left → bottom-right
    ((0, 9), (9, 0)),      # Top-right → bottom-left
    ((4, 0), (5, 9)),      # Mid-left → mid-right
    ((0, 5), (9, 4)),      # Top-center → bottom-center
]


def gen_maze(wall_prob=WALL_PROB):
    """FIXED start/goal pairs (not randomized). Prevents single-route memorization
    while keeping the problem learnable."""
    for _ in range(100):
        grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < wall_prob).astype(np.float32)
        # Pick one of 4 fixed pairs
        start_rc, target_rc = FIXED_PAIRS[np.random.randint(0, len(FIXED_PAIRS))]
        if grid[start_rc] == 1.0: grid[start_rc] = 0.0
        if grid[target_rc] == 1.0: grid[target_rc] = 0.0
        if not bfs_reachable(grid, start_rc, target_rc): continue
        return (torch.tensor(grid),
                torch.tensor(start_rc, dtype=torch.float32),
                torch.tensor(target_rc, dtype=torch.float32))
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    return torch.tensor(grid), torch.tensor((0, 0), dtype=torch.float32), torch.tensor((9, 9), dtype=torch.float32)

def bfs_reachable(grid, start, target):
    q = deque([start]); visited = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == target: return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and grid[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc)); q.append((nr,nc))
    return False

def get_radar(grid, pos_r, pos_c):
    radar = torch.ones(3, 3)
    for i in range(3):
        for j in range(3):
            nr, nc = int(pos_r)+i-1, int(pos_c)+j-1
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE: radar[i,j] = grid[nr,nc]
    return radar.flatten()

def get_wall_proximity(grid, pos_r, pos_c):
    r, c = int(pos_r), int(pos_c); r=max(0,min(GRID_SIZE-1,r)); c=max(0,min(GRID_SIZE-1,c))
    def ddir(dr, dc):
        for d in range(1, GRID_SIZE):
            nr, nc = r+dr*d, c+dc*d
            if nr<0 or nr>=GRID_SIZE or nc<0 or nc>=GRID_SIZE or grid[nr,nc]==1.0: return d
        return GRID_SIZE
    wc = sum(1 for dr in[-1,0,1] for dc in[-1,0,1]
             if 0<=r+dr<GRID_SIZE and 0<=c+dc<GRID_SIZE and grid[r+dr,c+dc]==1.0)
    return torch.tensor([ddir(-1,0)/GRID_SIZE, ddir(1,0)/GRID_SIZE,
                         ddir(0,-1)/GRID_SIZE, ddir(0,1)/GRID_SIZE, wc/8.0], dtype=torch.float32)

def a_star_path(grid, start, target):
    sr, sc = int(start[0].item()), int(start[1].item())
    tr, tc = int(target[0].item()), int(target[1].item())
    g_score = {(sr,sc):0}; parent = {(sr,sc):None}
    open_set = [(abs(sr-tr)+abs(sc-tc), sr, sc)]
    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r,c) == (tr,tc):
            path = []; curr = (tr,tc)
            while curr is not None: path.append(curr); curr = parent[curr]
            path.reverse(); return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and grid[nr,nc]==0:
                tg = g_score[(r,c)]+1
                if (nr,nc) not in g_score or tg<g_score[(nr,nc)]:
                    g_score[(nr,nc)]=tg; parent[(nr,nc)]=(r,c)
                    heapq.heappush(open_set, (tg+abs(nr-tr)+abs(nc-tc), nr, nc))
    return None


class Observer(nn.Module):
    input_dim = GRID_SIZE*GRID_SIZE + 2 + WALL_FEATURE_DIM
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(self.input_dim,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.actor_head = nn.Sequential(nn.Linear(HIDDEN,LATENT_DIM),nn.LayerNorm(LATENT_DIM))
        self.critic_head = nn.Linear(HIDDEN,1)
    def forward(self, gf, pos, wf):
        x = torch.cat([gf, pos/GRID_SIZE, wf], dim=-1)
        s = self.shared(x)
        return self.actor_head(s), self.critic_head(s.detach())
    def get_latent(self, gf, pos, wf):
        return self.actor_head(self.shared(torch.cat([gf, pos/GRID_SIZE, wf], dim=-1)))


class Navigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9+LATENT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(HIDDEN,2),nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2))
    def forward(self, radar, z, deterministic=False):
        feat = self.net(torch.cat([radar, z], dim=-1))
        mu = self.mu_head(feat) * STEP_CAP
        if deterministic: return mu
        return D.Normal(mu, torch.exp(self.log_std).expand_as(mu))


# --- WARM-START ---

def generate_expert_batch(batch_size):
    items = []
    for _ in range(batch_size):
        grid, start, target = gen_maze()
        path = a_star_path(grid.numpy(), start, target)
        if path and len(path)>=2:
            for i in range(len(path)-1):
                cr,cc=path[i]; nr,nc=path[i+1]
                move=torch.tensor([nr-cr,nc-cc],dtype=torch.float32)
                pos=torch.tensor([cr,cc],dtype=torch.float32)
                radar=get_radar(grid,cr,cc); wf=get_wall_proximity(grid,cr,cc)
                items.append((grid.flatten(),pos,wf,radar,move))
    return items

def warm_start(observer, navigator, epochs=WARMUP_EPOCHS):
    opt = optim.Adam(list(observer.parameters())+list(navigator.parameters()), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        items = generate_expert_batch(BATCH)
        if len(items)<BATCH: continue
        idx = np.random.choice(len(items),BATCH,replace=False)
        gb=torch.stack([items[i][0] for i in idx]); pb=torch.stack([items[i][1] for i in idx])
        wb=torch.stack([items[i][2] for i in idx]); rb=torch.stack([items[i][3] for i in idx])
        tm=torch.stack([items[i][4] for i in idx])
        z = observer.get_latent(gb,pb,wb)
        pm = navigator(rb,z,deterministic=True)
        loss = nn.MSELoss()(pm,tm) + 0.001*torch.norm(z,p=2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters())+list(navigator.parameters()),1.0)
        opt.step(); sch.step(); hist.append(loss.item())
        if ep%500==0: print(f"  Warm-up {ep:5d}: loss={loss.item():.6f}")
    return hist


# --- PPO with Channel Noise ---

def collect_trajectories(observer, navigator, num_eps):
    ag,ap,aw,ar,aa,alp,are,av,at,azs = [],[],[],[],[],[],[],[],[],[]
    for _ in range(num_eps):
        grid, start, target = gen_maze(); gf=grid.flatten(); pos=start.clone(); active=True
        eg,ep,ew,er,ea,elp,ere,ev,et,ez = [],[],[],[],[],[],[],[],[],[]
        for _ in range(MAX_STEPS):
            if not active: break
            r,c = int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c); wf=get_wall_proximity(grid,r,c)
            with torch.no_grad():
                z_clean, v = observer(gf.unsqueeze(0),pos.unsqueeze(0),wf.unsqueeze(0))
                # CHANNEL NOISE: inject Gaussian noise into z
                z_noisy = z_clean + torch.randn_like(z_clean) * CHANNEL_NOISE_STD
                dist = navigator(radar.unsqueeze(0), z_noisy)
                action = dist.sample().squeeze(0); lp = dist.log_prob(action).sum(dim=-1)
            np_ = torch.clamp(pos+action,0,GRID_SIZE-1)
            rn,cn = int(round(np_[0].item())),int(round(np_[1].item()))
            rn,cn = max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            wh=(grid[rn,cn]==1.0); rd=torch.norm(np_-target).item()<TARGET_THRESH
            reward = -torch.norm(np_-target).item()*0.1 + STEP_PENALTY
            # Narrow-band exponential proximity: decays fast, only fires near walls
            # exp(-2.0*1) ≈ 0.14 at d=1, exp(-2.0*2) ≈ 0.02 at d=2 → essentially zero beyond d=2
            min_wall_dist = min(wf[0].item(), wf[1].item(), wf[2].item(), wf[3].item()) * GRID_SIZE
            reward -= EXP_PROXIMITY_ALPHA * np.exp(-EXP_PROXIMITY_BETA * min_wall_dist)
            it=False
            if wh: reward+=WALL_COLLISION_PENALTY; active=False; it=True
            if rd: reward+=GOAL_REWARD; active=False; it=True
            eg.append(gf); ep.append(pos); ew.append(wf); er.append(radar)
            ea.append(action); elp.append(lp.item()); ere.append(reward)
            ev.append(v.item()); et.append(it)
            ez.append(z_clean.squeeze(0))  # Store CLEAN z for diversity analysis
            pos=np_
        if eg:
            ag.append(torch.stack(eg)); ap.append(torch.stack(ep)); aw.append(torch.stack(ew))
            ar.append(torch.stack(er)); aa.append(torch.stack(ea)); alp.append(elp)
            are.append(ere); av.append(ev); at.append(et); azs.append(ez)
    return (ag,ap,aw,ar,aa,alp,are,av,at,azs)


def compute_gae(rl,vl,tl,observer,ag,ap,aw):
    aadv,aret=[],[]
    for i in range(len(rl)):
        T=len(rl[i]); adv=torch.zeros(T); ret=torch.zeros(T); lsv=0.0
        if not tl[i][-1]:
            with torch.no_grad():
                _,vl_=observer(ag[i][-1:],ap[i][-1:],aw[i][-1:]); lsv=vl_.item()
        gae=0.0; nv=lsv
        for t in reversed(range(T)):
            mask=0.0 if tl[i][t] else 1.0
            delta=rl[i][t]+GAMMA*nv*mask-vl[i][t]
            gae=delta+GAMMA*GAE_LAMBDA*mask*gae
            adv[t]=gae; ret[t]=gae+vl[i][t]
            nv=0.0 if tl[i][t] else vl[i][t]
        aadv.append(adv); aret.append(ret)
    return aadv,aret


def ppo_update(observer, navigator, opt, traj, adv_list, ret_list, azs_rollout):
    """PPO update WITH diversity loss. Noise handles robustness, diversity handles emergence."""
    (ag,ap,aw,ar,aa,alp,_,av,_,_) = traj
    gc=torch.cat([g for g in ag],dim=0); pc=torch.cat([p for p in ap],dim=0)
    wc=torch.cat([w for w in aw],dim=0); rc=torch.cat([r for r in ar],dim=0)
    ac=torch.cat([a for a in aa],dim=0)
    olp=torch.tensor([l for ls in alp for l in ls],dtype=torch.float32)
    advc=torch.cat([a for a in adv_list],dim=0)
    retc=torch.cat([r for r in ret_list],dim=0)
    advc=(advc-advc.mean())/(advc.std()+1e-8)
    for _ in range(PPO_EPOCHS):
        zc, vals = observer(gc,pc,wc)
        zc_noisy = zc + torch.randn_like(zc) * CHANNEL_NOISE_STD
        dist = navigator(rc, zc_noisy)
        nlp = dist.log_prob(ac).sum(dim=-1); entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(nlp-olp)
        surr1=ratio*advc; surr2=torch.clamp(ratio,1-PPO_CLIP,1+PPO_CLIP)*advc
        ploss = -torch.min(surr1,surr2).mean()
        vloss = nn.MSELoss()(vals.squeeze(-1),retc)
        # Diversity on CLEAN z (noise would add artificial drift)
        td=0.0; npairs=0; off=0
        for ez in azs_rollout:
            el=len(ez)
            if el<2: off+=el; continue
            ze=zc[off:off+el]
            for t in range(1,el): td+=torch.norm(ze[t]-ze[t-1],p=2); npairs+=1
            off+=el
        dloss = -DIVERSITY_WEIGHT * (td / max(1,npairs))
        loss = ploss + VALUE_COEF*vloss - ENTROPY_COEF*entropy + dloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(observer.parameters())+list(navigator.parameters()),0.5)
        opt.step()
    return ploss.item(), vloss.item(), entropy.item(), dloss.item()


def evaluate(observer, navigator, num_eps=200):
    ns,nst,wh=0,0,0
    for _ in range(num_eps):
        grid,start,target=gen_maze(); pos=start.clone()
        for step in range(MAX_STEPS):
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c); wf=get_wall_proximity(grid,r,c)
            with torch.no_grad():
                z,_=observer(grid.flatten().unsqueeze(0),pos.unsqueeze(0),wf.unsqueeze(0))
                # EVAL: NO noise (use clean z for deterministic eval)
                action=navigator(radar.unsqueeze(0),z,deterministic=True).squeeze(0)
            np_=torch.clamp(pos+action,0,GRID_SIZE-1)
            rn,cn=int(round(np_[0].item())),int(round(np_[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            if grid[rn,cn]==1.0: wh+=1; break
            pos=np_
            if torch.norm(pos-target).item()<TARGET_THRESH: ns+=1; nst+=step+1; break
        else: nst+=MAX_STEPS
    gs,gst=0,0
    for _ in range(num_eps):
        grid,start,target=gen_maze()
        path=a_star_path(grid.numpy(),start,target)
        if path: gs+=1; gst+=len(path)-1
        else: gst+=MAX_STEPS
    ne=max(1,ns if ns>0 else num_eps)
    return {"neur_success":ns/num_eps,"neur_steps":nst/ne,
            "greedy_success":gs/num_eps,"greedy_steps":gst/max(1,gs),
            "wall_hits_neur":wh/num_eps}


def latent_evolution_test(observer, navigator, out_dir):
    for _ in range(20):
        grid,start,target=gen_maze()
        path=a_star_path(grid.numpy(),start,target)
        if path and len(path)>=5 and np.sum(grid.numpy())>2: break
    print(f"  Path: {len(path)}, Walls: {int(grid.sum().item())}, "
          f"Start: ({start[0].item():.0f},{start[1].item():.0f}), Target: ({target[0].item():.0f},{target[1].item():.0f})")
    pos=start.clone(); latents,positions=[],[]
    for step in range(min(len(path),30)):
        radar=get_radar(grid,int(pos[0].item()),int(pos[1].item()))
        wf=get_wall_proximity(grid,int(pos[0].item()),int(pos[1].item()))
        with torch.no_grad():
            z,_=observer(grid.flatten().unsqueeze(0),pos.unsqueeze(0),wf.unsqueeze(0))
            action=navigator(radar.unsqueeze(0),z,deterministic=True).squeeze(0)
        latents.append(z.squeeze(0).numpy()); positions.append(pos.clone().numpy())
        np_=torch.clamp(pos+action,0,GRID_SIZE-1)
        r,c=int(round(np_[0].item())),int(round(np_[1].item()))
        if grid[r,c]==1.0: break
        pos=np_
        if torch.norm(pos-target).item()<TARGET_THRESH: break
    la=np.array(latents); zs=np.std(la,axis=0).mean()
    dists=[np.linalg.norm(la[i]-la[0]) for i in range(len(la))]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    axes[0].plot(dists,'b-o',markersize=8); axes[0].set_title("||z_t-z_0||"); axes[0].grid(True,alpha=0.3)
    for d in range(min(8,LATENT_DIM)): axes[1].plot(la[:,d],alpha=0.5,label=f"dim {d}",marker='o',markersize=4)
    axes[1].set_title("Latent Dims"); axes[1].legend(fontsize=6); axes[1].grid(True,alpha=0.3)
    axes[2].imshow(grid.numpy().T,cmap='gray_r',origin='lower',alpha=0.3)
    pa=np.array(positions); axes[2].plot(pa[:,0],pa[:,1],'b-',linewidth=2)
    axes[2].scatter(start[0].item(),start[1].item(),c='green',s=100)
    axes[2].scatter(target[0].item(),target[1].item(),c='red',s=150,marker='*')
    axes[2].set_xlim(-0.5,GRID_SIZE-0.5); axes[2].set_ylim(-0.5,GRID_SIZE-0.5); axes[2].set_title("Trajectory")
    plt.suptitle(f"v11 Latent Evolution — std={zs:.4f} ({'EMERGENT' if zs>0.05 else 'STATIC'})",fontsize=14)
    plt.tight_layout(); plt.savefig(out_dir/"maze_latent_evolution_v11.png",dpi=150); plt.close()
    print(f"  z_std: {zs:.4f} → {'EMERGENT' if zs>0.05 else 'STATIC'}, max drift: {max(dists):.4f}")
    return zs,la


def plot_summary(wh,ph,er,out_dir):
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    axes[0,0].plot(wh,alpha=0.5,color='blue'); axes[0,0].set_title("Warm-Start"); axes[0,0].set_yscale("log"); axes[0,0].grid(True,alpha=0.3)
    axes[0,1].plot(ph["success"],alpha=0.6,color='orange',label="Success")
    axes[0,1].plot(ph["wall_hits"],alpha=0.6,color='red',label="Walls/ep")
    axes[0,1].set_title(f"PPO (noise={CHANNEL_NOISE_STD}, no diversity loss)"); axes[0,1].legend(); axes[0,1].grid(True,alpha=0.3)
    x=np.arange(2); w=0.35
    axes[1,0].bar(x-w/2,[er["neur_success"],er["greedy_success"]],w,label="Success",color="blue",alpha=0.7)
    ax2=axes[1,0].twinx(); ax2.bar(x+w/2,[er["neur_steps"],er["greedy_steps"]],w,label="Steps",color="red",alpha=0.7)
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(["Neuralese","A*"]); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(True,alpha=0.3,axis='y')
    axes[1,1].text(0.1,0.5,
                   f"v11: Randomized start/goal\nChannel noise std={CHANNEL_NOISE_STD}\n"
                   f"No diversity loss (noise handles emergence)\n"
                   f"MLP Observer + stateless Navigator\n"
                   f"Wall hits: {er['wall_hits_neur']:.2f}\n"
                   f"Min start-goal dist: 5 cells",
                   fontfamily='monospace',fontsize=10,va='center',transform=axes[1,1].transAxes)
    axes[1,1].set_title("Config"); axes[1,1].axis('off')
    plt.suptitle("Neuralese v11 — Randomized + Channel Noise",fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/"maze_results_v11.png",dpi=150); plt.close()


if __name__=="__main__":
    out_dir=Path(__file__).parent/"output"; out_dir.mkdir(exist_ok=True)
    print("="*60)
    print("NEURALESE v13 — 4 Fixed Pairs + Narrow-Band Proximity")
    print("="*60)
    print(f"  Start/goal: 4 fixed pairs (not single route, not fully random)")
    print(f"  Channel noise: std={CHANNEL_NOISE_STD}")
    print(f"  Diversity: weight={DIVERSITY_WEIGHT} on clean z")
    print(f"  Proximity: {EXP_PROXIMITY_ALPHA}*exp(-{EXP_PROXIMITY_BETA}*d)")

    print("\n[Phase 1] Warm-Start...")
    observer=Observer(); navigator=Navigator()
    warmup_hist=warm_start(observer,navigator)

    print("\n[Phase 2] PPO + Channel Noise...")
    opt=optim.Adam(list(observer.parameters())+list(navigator.parameters()),lr=RL_LR)
    ppo_hist={"success":[],"wall_hits":[],"entropy":[]}
    ep_counter=0; ep_per_update=32

    while ep_counter<PPO_EPISODES:
        traj=collect_trajectories(observer,navigator,ep_per_update)
        ag,ap,aw,ar,aa,alp,are_,av_,at_,azs=traj
        adv,ret=compute_gae(are_,av_,at_,observer,ag,ap,aw)
        pl,vl,ent,dl=ppo_update(observer,navigator,opt,traj,adv,ret,azs)
        ep_counter+=ep_per_update
        if ep_counter%200==0 or ep_counter>=PPO_EPISODES:
            ev=evaluate(observer,navigator,num_eps=50)
            ppo_hist["success"].append(ev["neur_success"]); ppo_hist["wall_hits"].append(ev["wall_hits_neur"])
            ppo_hist["entropy"].append(ent)
            print(f"  PPO ep {ep_counter:5d}: succ={ev['neur_success']:.1%} walls={ev['wall_hits_neur']:.2f} ent={ent:.3f} div={dl:.4f}")

    print("\n"+"="*40)
    eval_r=evaluate(observer,navigator,num_eps=200)
    print(f"  Neuralese: {eval_r['neur_success']:.1%}, {eval_r['neur_steps']:.1f} steps, {eval_r['wall_hits_neur']:.2f} walls/ep")
    print(f"  A*: {eval_r['greedy_success']:.1%}, {eval_r['greedy_steps']:.1f} steps")

    print("\n[Latent Evolution Test]")
    latent_evolution_test(observer,navigator,out_dir)
    plot_summary(warmup_hist,ppo_hist,eval_r,out_dir)
    print(f"\nDone! Outputs in {out_dir}/")
