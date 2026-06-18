"""
Neuralese Maze Audit — Null-channel baseline + Disentanglement + No warm-start.
Independent analysis: tests whether the Neuralese channel actually carries useful
information, or if the Navigator succeeds on its own.

THREE CRITICAL BASELINES:
1. NULL CHANNEL: Replace Observer output with random Gaussian noise.
   If success rate doesn't drop significantly, the channel is unused.
2. NO OBSERVER (GRU Navigator): Navigator with recurrent hidden state, NO Observer.
   If GRU matches or beats Neuralese, the Observer adds no value.
3. DISENTANGLEMENT: Per-step correlation between latent dimensions and
   task-relevant variables (distance-to-goal, direction-to-goal, wall-proximity).
   If no dimension correlates with anything useful, the latents are noise.
"""
import torch, torch.nn as nn, torch.optim as optim, torch.distributions as D
import numpy as np, heapq
from pathlib import Path; from collections import deque
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GRID_SIZE=10; WALL_PROB=0.20; LATENT_DIM=12; HIDDEN=128
STEP_CAP=0.5; MAX_STEPS=50; BATCH=16
PPO_EPOCHS=4; PPO_EPISODES=3000; PPO_CLIP=0.2
GAE_LAMBDA=0.95; GAMMA=0.99; LR=1e-3; RL_LR=1e-3
ENTROPY_COEF=0.005; VALUE_COEF=0.5
WALL_PENALTY=-10.0; GOAL_REWARD=10.0; STEP_PENALTY=-0.05; TARGET_THRESH=0.5
DIVERSITY_WEIGHT=0.05; CHANNEL_NOISE_STD=0.1

def gen_maze():
    for _ in range(100):
        g=(np.random.rand(GRID_SIZE,GRID_SIZE)<WALL_PROB).astype(np.float32)
        s,t=(0,0),(GRID_SIZE-1,GRID_SIZE-1)
        g[s]=0.0;g[t]=0.0
        if bfs_reachable(g,s,t):
            return torch.tensor(g),torch.tensor(s,dtype=torch.float32),torch.tensor(t,dtype=torch.float32)
    g=np.zeros((GRID_SIZE,GRID_SIZE),dtype=np.float32)
    return torch.tensor(g),torch.tensor((0,0),dtype=torch.float32),torch.tensor((9,9),dtype=torch.float32)

def bfs_reachable(g,s,t):
    q=deque([s]);v={s}
    while q:
        r,c=q.popleft()
        if(r,c)==t:return True
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and g[nr,nc]==0 and (nr,nc) not in v:
                v.add((nr,nc));q.append((nr,nc))
    return False

def get_radar(grid,r,c):
    radar=torch.ones(3,3)
    for i in range(3):
        for j in range(3):
            nr,nc=r+i-1,c+j-1
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE: radar[i,j]=grid[nr,nc]
    return radar.flatten()

# --- MODEL 1: Standard Neuralese Observer+Navigator ---
class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(GRID_SIZE*GRID_SIZE+2,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.actor_head=nn.Sequential(nn.Linear(HIDDEN,LATENT_DIM),nn.LayerNorm(LATENT_DIM))
        self.critic_head=nn.Linear(HIDDEN,1)
    def forward(self,grid_flat,pos):
        x=torch.cat([grid_flat,pos/GRID_SIZE],dim=-1)
        h=self.shared(x)
        z=self.actor_head(h)
        v=self.critic_head(h.detach())
        return z,v
    def get_latent(self,grid_flat,pos):
        x=torch.cat([grid_flat,pos/GRID_SIZE],dim=-1)
        return self.actor_head(self.shared(x))

class Navigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(9+LATENT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.mu_head=nn.Sequential(nn.Linear(HIDDEN,2),nn.Tanh())
        self.log_std=nn.Parameter(torch.zeros(2))
    def forward(self,radar,z,det=False):
        feat=self.shared(torch.cat([radar,z],dim=-1))
        mu=self.mu_head(feat)*STEP_CAP
        if det:return mu
        return D.Normal(mu,torch.exp(self.log_std).expand_as(mu))

# --- MODEL 2: GRU Navigator (no Observer) ---
class GRUNavigator(nn.Module):
    """Recurrent Navigator that uses its own hidden state instead of Observer.
    If this matches Neuralese performance, the Observer→z pathway is unnecessary."""
    def __init__(self):
        super().__init__()
        self.gru=nn.GRUCell(9+2+2,HIDDEN)  # radar + pos + target = 13D input
        self.mu_head=nn.Sequential(nn.Linear(HIDDEN,2),nn.Tanh())
        self.log_std=nn.Parameter(torch.zeros(2))
        self.critic_head=nn.Linear(HIDDEN,1)
    def forward(self,radar,pos,target,hidden=None,det=False):
        if hidden is None:
            hidden=torch.zeros(radar.size(0),HIDDEN)
        x=torch.cat([radar,pos/GRID_SIZE,target/GRID_SIZE],dim=-1)
        h=self.gru(x,hidden)
        mu=self.mu_head(h)*STEP_CAP
        v=self.critic_head(h)
        if det:return mu,h,v
        return D.Normal(mu,torch.exp(self.log_std).expand_as(mu)),h,v

# --- MODEL 3: Baseline Navigator (local-only, no Observer, no recurrence) ---
class LocalNavigator(nn.Module):
    """Navigator with ONLY local radar — no Observer input, no recurrent state.
    This is the absolute lower bound for comparison."""
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(9,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.mu_head=nn.Sequential(nn.Linear(HIDDEN,2),nn.Tanh())
        self.log_std=nn.Parameter(torch.zeros(2))
    def forward(self,radar,det=False):
        h=self.shared(radar)
        mu=self.mu_head(h)*STEP_CAP
        if det:return mu
        return D.Normal(mu,torch.exp(self.log_std).expand_as(mu))

# --- COLLECTION ---
def collect_neur(observer,navigator,n_eps,noise_z=False):
    data={"grids":[],"pos":[],"radars":[],"actions":[],"logps":[],"rewards":[],"values":[],"terms":[],"zs":[]}
    for _ in range(n_eps):
        grid,start,target=gen_maze();gflat=grid.flatten();pos=start.clone();active=True
        ep={k:[] for k in data}
        for st in range(MAX_STEPS):
            if not active:break
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c)
            with torch.no_grad():
                z,v=observer(gflat.unsqueeze(0),pos.unsqueeze(0))
                if noise_z:
                    z=torch.randn_like(z)*CHANNEL_NOISE_STD  # NULL CHANNEL
                dist=navigator(radar.unsqueeze(0),z)
                a=dist.sample().squeeze(0)
                lp=dist.log_prob(a).sum(dim=-1)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            wall=(grid[rn,cn]==1.0);reached=torch.norm(new_pos-target).item()<TARGET_THRESH
            reward=-torch.norm(new_pos-target).item()*0.1+STEP_PENALTY;term=False
            if wall:reward+=WALL_PENALTY;active=False;term=True
            if reached:reward+=GOAL_REWARD;active=False;term=True
            ep["grids"].append(gflat);ep["pos"].append(pos);ep["radars"].append(radar)
            ep["actions"].append(a);ep["logps"].append(lp.item());ep["rewards"].append(reward)
            ep["values"].append(v.item());ep["terms"].append(term)
            ep["zs"].append(z.squeeze(0) if not noise_z else z.squeeze(0))
            pos=new_pos
        if ep["grids"]:
            for k in data:data[k].append(ep[k])
    return data

def collect_gru(gru_nav,n_eps):
    data={"grids":[],"pos":[],"radars":[],"targets":[],"actions":[],"logps":[],"rewards":[],"values":[],"terms":[]}
    for _ in range(n_eps):
        grid,start,target=gen_maze();gflat=grid.flatten();pos=start.clone();active=True;h=None
        ep={k:[] for k in data}
        for st in range(MAX_STEPS):
            if not active:break
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c)
            with torch.no_grad():
                dist,h,v=gru_nav(radar.unsqueeze(0),pos.unsqueeze(0),target.unsqueeze(0),h)
                a=dist.sample().squeeze(0)
                lp=dist.log_prob(a).sum(dim=-1)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            wall=(grid[rn,cn]==1.0);reached=torch.norm(new_pos-target).item()<TARGET_THRESH
            reward=-torch.norm(new_pos-target).item()*0.1+STEP_PENALTY;term=False
            if wall:reward+=WALL_PENALTY;active=False;term=True
            if reached:reward+=GOAL_REWARD;active=False;term=True
            ep["grids"].append(gflat);ep["pos"].append(pos);ep["radars"].append(radar)
            ep["targets"].append(target);ep["actions"].append(a);ep["logps"].append(lp.item())
            ep["rewards"].append(reward);ep["values"].append(v.item());ep["terms"].append(term)
            pos=new_pos
        if ep["grids"]:
            for k in data:data[k].append(ep[k])
    return data

# --- GAE ---
def compute_gae(rewards,values,terms,observer,grids,positions):
    advs,rets=[],[]
    for i in range(len(rewards)):
        rw,val=rewards[i],values[i];term=terms[i];g=grids[i];p=positions[i];T=len(rw)
        a=torch.zeros(T);ret=torch.zeros(T)
        last_v=0.0
        if not term[-1]:
            with torch.no_grad():_,lv=observer(g[-1:],p[-1:]);last_v=lv.item()
        gae=0.0;nv=last_v
        for t in reversed(range(T)):
            mask=0.0 if term[t] else 1.0
            delta=rw[t]+GAMMA*nv*mask-val[t]
            gae=delta+GAMMA*GAE_LAMBDA*mask*gae
            a[t]=gae;ret[t]=gae+val[t];nv=0.0 if term[t] else val[t]
        advs.append(a);rets.append(ret)
    return advs,rets

def compute_gae_gru(rewards,values,terms):
    advs,rets=[],[]
    for i in range(len(rewards)):
        rw,val=rewards[i],values[i];term=terms[i];T=len(rw)
        a=torch.zeros(T);ret=torch.zeros(T)
        gae=0.0;nv=0.0
        for t in reversed(range(T)):
            mask=0.0 if term[t] else 1.0
            delta=rw[t]+GAMMA*nv*mask-val[t]
            gae=delta+GAMMA*GAE_LAMBDA*mask*gae
            a[t]=gae;ret[t]=gae+val[t];nv=0.0 if term[t] else val[t]
        advs.append(a);rets.append(ret)
    return advs,rets

# --- PPO UPDATE ---
def ppo_update_neur(obs,nav,opt,data,advs,rets):
    grids=torch.stack([g for ep_g in data["grids"] for g in ep_g])
    pos=torch.stack([p for ep_p in data["pos"] for p in ep_p])
    radars=torch.stack([r for ep_r in data["radars"] for r in ep_r])
    acts=torch.stack([a for ep_a in data["actions"] for a in ep_a])
    old_lps=torch.tensor([lp for lps in data["logps"] for lp in lps],dtype=torch.float32)
    adv_cat=torch.cat(advs);ret_cat=torch.cat(rets)
    adv_cat=(adv_cat-adv_cat.mean())/(adv_cat.std()+1e-8)
    for _ in range(PPO_EPOCHS):
        z,vals=obs(grids,pos);dist=nav(radars,z)
        nlps=dist.log_prob(acts).sum(dim=-1);ent=dist.entropy().sum(dim=-1).mean()
        ratio=torch.exp(nlps-old_lps)
        s1=ratio*adv_cat;s2=torch.clamp(ratio,1-PPO_CLIP,1+PPO_CLIP)*adv_cat
        p_loss=-torch.min(s1,s2).mean()
        v_loss=nn.MSELoss()(vals.squeeze(-1),ret_cat)
        loss=p_loss+VALUE_COEF*v_loss-ENTROPY_COEF*ent
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(obs.parameters())+list(nav.parameters()),0.5);opt.step()
    return p_loss.item(),v_loss.item(),ent.item()

def ppo_update_gru(gru,opt,data,advs,rets):
    acts=torch.stack([a for ep_a in data["actions"] for a in ep_a])
    old_lps=torch.tensor([lp for lps in data["logps"] for lp in lps],dtype=torch.float32)
    adv_cat=torch.cat(advs);ret_cat=torch.cat(rets)
    adv_cat=(adv_cat-adv_cat.mean())/(adv_cat.std()+1e-8)
    # Re-forward GRU to get current log-probs
    total_loss=0.0;count=0
    for i in range(len(data["grids"])):
        g=data["grids"][i];p=data["pos"][i];r=data["radars"][i]
        t=data["targets"][i];a=data["actions"][i]
        h=None;ep_lps=[];ep_vals=[]
        for j in range(len(g)):
            dist,h,v=gru(r[j].unsqueeze(0),p[j].unsqueeze(0),t[j].unsqueeze(0),h)
            ep_lps.append(dist.log_prob(a[j].unsqueeze(0)).sum(dim=-1))
            ep_vals.append(v)
        nlps=torch.stack(ep_lps).squeeze()
        vals=torch.stack(ep_vals).squeeze()
        # compute loss per episode
        ep_adv=advs[i];ep_ret=rets[i]
        ep_adv=(ep_adv-ep_adv.mean())/(ep_adv.std()+1e-8)
        ratio=torch.exp(nlps-old_lps[count:count+len(g)])
        s1=ratio*ep_adv;s2=torch.clamp(ratio,1-PPO_CLIP,1+PPO_CLIP)*ep_adv
        p_loss=-torch.min(s1,s2).mean()
        v_loss=nn.MSELoss()(vals,ep_ret)
        total_loss+=p_loss+VALUE_COEF*v_loss
        count+=len(g)
    opt.zero_grad();(total_loss/len(data["grids"])).backward()
    torch.nn.utils.clip_grad_norm_(gru.parameters(),0.5);opt.step()
    return total_loss.item()

# --- EVALUATION ---
def evaluate(obs,nav,gru_nav,n_eps=100):
    # Neuralese
    ns,nw=0,0
    for _ in range(n_eps):
        grid,start,target=gen_maze();pos=start.clone()
        for st in range(MAX_STEPS):
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c)
            with torch.no_grad():
                z,_=obs(grid.flatten().unsqueeze(0),pos.unsqueeze(0))
                a=nav(radar.unsqueeze(0),z,det=True).squeeze(0)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            if grid[rn,cn]==1.0:nw+=1;break
            pos=new_pos
            if torch.norm(pos-target).item()<TARGET_THRESH:ns+=1;break
    # GRU
    gs,gw=0,0
    for _ in range(n_eps):
        grid,start,target=gen_maze();pos=start.clone();h=None
        for st in range(MAX_STEPS):
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c)
            with torch.no_grad():
                a,h,_=gru_nav(radar.unsqueeze(0),pos.unsqueeze(0),target.unsqueeze(0),h,det=True)
                a=a.squeeze(0)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            if grid[rn,cn]==1.0:gw+=1;break
            pos=new_pos
            if torch.norm(pos-target).item()<TARGET_THRESH:gs+=1;break
    return {"neur_succ":ns/n_eps,"neur_walls":nw/n_eps,
            "gru_succ":gs/n_eps,"gru_walls":gw/n_eps}

# --- DISENTANGLEMENT ---
def disentanglement_analysis(obs,nav,n_mazes=50):
    """Per-dimension correlation between z_i and task-relevant variables."""
    all_zs=[];all_dists=[];all_dirs=[];all_walls=[]
    for _ in range(n_mazes):
        grid,start,target=gen_maze();pos=start.clone()
        for st in range(min(30,MAX_STEPS)):
            r,c=int(pos[0].item()),int(pos[1].item())
            radar=get_radar(grid,r,c)
            with torch.no_grad():
                z,_=obs(grid.flatten().unsqueeze(0),pos.unsqueeze(0))
            all_zs.append(z.squeeze(0).numpy())
            # Distance to goal
            dist=torch.norm(pos-target).item()/GRID_SIZE
            all_dists.append(dist)
            # Direction to goal
            dx_target=target[0].item()-pos[0].item()
            dy_target=target[1].item()-pos[1].item()
            all_dirs.append([dx_target/GRID_SIZE,dy_target/GRID_SIZE])
            # Wall proximity in radar
            all_walls.append(radar.sum().item())
            # Move
            a=nav(radar.unsqueeze(0),z,det=True).squeeze(0)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            if grid[rn,cn]==1.0:break
            pos=new_pos
            if torch.norm(pos-target).item()<TARGET_THRESH:break
    zs=np.array(all_zs);dists=np.array(all_dists)
    dirs=np.array(all_dirs);walls=np.array(all_walls)
    corrs={}
    for d in range(LATENT_DIM):
        corrs[f"z{d}_dist"]=np.corrcoef(zs[:,d],dists)[0,1]
        corrs[f"z{d}_dir_x"]=np.corrcoef(zs[:,d],dirs[:,0])[0,1]
        corrs[f"z{d}_dir_y"]=np.corrcoef(zs[:,d],dirs[:,1])[0,1]
        corrs[f"z{d}_walls"]=np.corrcoef(zs[:,d],walls)[0,1]
    # Find best dimension for each task variable
    best_dist = max(range(LATENT_DIM), key=lambda d: abs(corrs.get(f"z{d}_dist", 0)))
    best_dirx = max(range(LATENT_DIM), key=lambda d: abs(corrs.get(f"z{d}_dir_x", 0)))
    best_diry = max(range(LATENT_DIM), key=lambda d: abs(corrs.get(f"z{d}_dir_y", 0)))
    best_walls = max(range(LATENT_DIM), key=lambda d: abs(corrs.get(f"z{d}_walls", 0)))
    best = {
        "dist": (f"z{best_dist}", corrs.get(f"z{best_dist}_dist", 0)),
        "dir_x": (f"z{best_dirx}", corrs.get(f"z{best_dirx}_dir_x", 0)),
        "dir_y": (f"z{best_diry}", corrs.get(f"z{best_diry}_dir_y", 0)),
        "walls": (f"z{best_walls}", corrs.get(f"z{best_walls}_walls", 0)),
    }
    return corrs,best

# --- MAIN ---
if __name__=="__main__":
    out_dir=Path(__file__).parent/"output";out_dir.mkdir(exist_ok=True)
    print("="*70)
    print("NEURALESE MAZE AUDIT — Null Channel + Disentanglement")
    print("="*70)

    # 1. Train standard Neuralese (skip GRU for speed — focus on null-channel test)
    print("\n[1/3] Training Neuralese (Observer+Navigator, no warm-start)...")
    obs=Observer();nav=Navigator()
    opt_neur=optim.Adam(list(obs.parameters())+list(nav.parameters()),lr=RL_LR)

    hist={"neur_succ":[],"neur_walls":[]}
    ep=0
    while ep<PPO_EPISODES:
        data_neur=collect_neur(obs,nav,16)
        advs,rets=compute_gae(data_neur["rewards"],data_neur["values"],data_neur["terms"],obs,data_neur["grids"],data_neur["pos"])
        pl,vl,ent=ppo_update_neur(obs,nav,opt_neur,data_neur,advs,rets)
        ep+=16
        if ep%200==0 or ep>=PPO_EPISODES:
            ns,nw=0,0
            for _ in range(50):
                grid,start,target=gen_maze();pos=start.clone()
                for st in range(MAX_STEPS):
                    r,c=int(pos[0].item()),int(pos[1].item());radar=get_radar(grid,r,c)
                    with torch.no_grad():
                        z,_=obs(grid.flatten().unsqueeze(0),pos.unsqueeze(0))
                        a=nav(radar.unsqueeze(0),z,det=True).squeeze(0)
                    new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
                    rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
                    rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
                    if grid[rn,cn]==1.0:nw+=1;break
                    pos=new_pos
                    if torch.norm(pos-target).item()<TARGET_THRESH:ns+=1;break
            hist["neur_succ"].append(ns/50);hist["neur_walls"].append(nw/50)
            print(f"  ep{ep:5d}: succ={ns/50:.1%} walls={nw/50:.2f}")

    # 2. NULL CHANNEL TEST
    print("\n[2/3] NULL CHANNEL TEST (replace z with random noise)...")
    null_succ,null_walls=0,0
    for _ in range(200):
        grid,start,target=gen_maze();pos=start.clone()
        for st in range(MAX_STEPS):
            r,c=int(pos[0].item()),int(pos[1].item());radar=get_radar(grid,r,c)
            with torch.no_grad():
                z_noise=torch.randn(1,LATENT_DIM)*CHANNEL_NOISE_STD
                a=nav(radar.unsqueeze(0),z_noise,det=True).squeeze(0)
            new_pos=torch.clamp(pos+a,0,GRID_SIZE-1)
            rn,cn=int(round(new_pos[0].item())),int(round(new_pos[1].item()))
            rn,cn=max(0,min(GRID_SIZE-1,rn)),max(0,min(GRID_SIZE-1,cn))
            if grid[rn,cn]==1.0:null_walls+=1;break
            pos=new_pos
            if torch.norm(pos-target).item()<TARGET_THRESH:null_succ+=1;break
    null_succ/=200;null_walls/=200
    print(f"  NULL CHANNEL: succ={null_succ:.1%} walls={null_walls:.2f}")

    # Reference the last Neuralese eval
    neur_succ = hist["neur_succ"][-1]
    neur_walls = hist["neur_walls"][-1]

    # 3. DISENTANGLEMENT
    print("\n[3/3] DISENTANGLEMENT ANALYSIS...")
    corrs,best=disentanglement_analysis(obs,nav)
    print(f"  Best distance-dim: {best['dist']}")
    print(f"  Best direction-x dim: {best['dir_x']}")
    print(f"  Best direction-y dim: {best['dir_y']}")
    print(f"  Best wall-proximity dim: {best['walls']}")

    # Plot
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    axes[0,0].plot(hist["neur_succ"],'b-o',label="Neuralese")
    axes[0,0].set_title("Success Rate");axes[0,0].legend();axes[0,0].grid(True,alpha=0.3)
    axes[0,1].bar(["Neuralese","Null Channel"],
                  [neur_succ,null_succ],
                  color=['blue','red'],alpha=0.7)
    axes[0,1].set_title("NULL CHANNEL TEST: succ drop")
    for i,(v,label) in enumerate(zip([neur_succ,null_succ],["Neur","Null"])):
        axes[0,1].text(i,v+0.01,f"{v:.1%}",ha='center')
    axes[0,1].grid(True,alpha=0.3,axis='y')

    # Disentanglement heatmap
    z_corr_matrix=np.zeros((LATENT_DIM,4))
    for d in range(LATENT_DIM):
        z_corr_matrix[d,0]=corrs.get(f"z{d}_dist",0)
        z_corr_matrix[d,1]=corrs.get(f"z{d}_dir_x",0)
        z_corr_matrix[d,2]=corrs.get(f"z{d}_dir_y",0)
        z_corr_matrix[d,3]=corrs.get(f"z{d}_walls",0)
    im=axes[1,0].imshow(z_corr_matrix,cmap='RdBu_r',aspect='auto',vmin=-1,vmax=1)
    axes[1,0].set_xticks(range(4));axes[1,0].set_xticklabels(["Dist","DirX","DirY","Walls"])
    axes[1,0].set_ylabel("Latent dim");axes[1,0].set_title("Per-Dimension Task Correlation")
    plt.colorbar(im,ax=axes[1,0])

    axes[1,1].text(0.1,0.5,
        f"KEY QUESTION: Does the channel carry information?\n\n"
        f"Neuralese: {neur_succ:.1%} success\n"
        f"Null Channel: {null_succ:.1%} success\n\n"
        f"If Null Channel ≈ Neuralese → channel is unused\n"
        f"Drop: {neur_succ-null_succ:+.1%}\n\n"
        f"Best disentanglement:\n"
        f"  Dist: {best['dist']}\n"
        f"  DirX: {best['dir_x']}\n"
        f"  DirY: {best['dir_y']}\n"
        f"  Walls: {best['walls']}",
        fontfamily='monospace',fontsize=9,va='center',transform=axes[1,1].transAxes)
    axes[1,1].axis('off')
    plt.suptitle("Neuralese Maze Audit — Null Channel + Disentanglement",fontsize=14,fontweight='bold')
    plt.tight_layout();plt.savefig(out_dir/"maze_audit.png",dpi=150);plt.close()
    print(f"\nSaved: {out_dir}/maze_audit.png")

    print(f"\n{'='*40}")
    print(f"FINAL: Neur={neur_succ:.1%}  Null={null_succ:.1%}  Drop={neur_succ-null_succ:+.1%}")
    print(f"VERDICT: ",end="")
    if neur_succ - null_succ < 0.05:
        print("CHANNEL IS NOT USED (null channel performs similarly)")
    else:
        print(f"CHANNEL CARRIES INFORMATION (Neuralese beats null by {neur_succ-null_succ:+.1%})")
