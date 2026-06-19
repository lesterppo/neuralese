#!/usr/bin/env python3
"""Neuralese Cross-Architecture — Self-contained test."""
import sys, torch, numpy as np, torch.nn as nn, torch.optim as optim, torch.distributions as D
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
LATENT_DIM = 8; HIDDEN = 128; EMBED_DIM = 64

# ---- Function Generation (standalone) ----
def gen_function():
    templates = [
        ("merge", ["d1","d2"], "return {**d1,**d2}"),
        ("filter_pos", ["items"], "return [x for x in items if x>0]"),
        ("flatten", ["nested"], "return [x for s in nested for x in s]"),
        ("group_by", ["recs","k"], "d={}; [d.setdefault(r[k],[]).append(r) for r in recs]; return d"),
        ("normalize", ["data"], "m=sum(data)/len(data); s=(sum((x-m)**2 for x in data)/len(data))**0.5; return [(x-m)/s for x in data]"),
        ("tokenize", ["text"], "return text.lower().split()"),
        ("camel", ["name"], "p=name.split('_'); return p[0]+''.join(x.title() for x in p[1:])"),
        ("snake", ["name"], "return ''.join('_'+c.lower() if c.isupper() else c for c in name).lstrip('_')"),
        ("truncate", ["s","n"], "return s[:n]+'...' if len(s)>n else s"),
        ("json_read", ["path"], "import json; with open(path) as f: return json.load(f)"),
        ("csv_write", ["rows","path"], "import csv; with open(path,'w') as f: csv.writer(f).writerows(rows)"),
        ("list_ext", ["d","e"], "import os; return [f for f in os.listdir(d) if f.endswith(e)]"),
        ("safe_rm", ["path"], "import os; os.remove(path) if os.path.exists(path) else None"),
        ("cache_get", ["c","k","ms"], "if k in c: c.move_to_end(k); return c[k]; return None"),
        ("bs", ["arr","x"], "lo,hi=0,len(arr)-1; while lo<=hi: m=(lo+hi)//2; v=arr[m]; return m if v==x else (lo:=m+1) if v<x else (hi:=m-1); return -1"),
        ("topk", ["items","k","kf"], "return sorted(items,key=kf,reverse=True)[:k]"),
        ("dedupe", ["items"], "seen=set(); return [x for x in items if not (x in seen or seen.add(x))]"),
        ("chunk", ["seq","n"], "return [seq[i:i+n] for i in range(0,len(seq),n)]"),
        ("freq", ["items"], "from collections import Counter; return dict(Counter(items))"),
        ("shuffle", ["lst"], "import random; r=lst[:]; random.shuffle(r); return r"),
        ("pick", ["lst","n"], "import random; return random.sample(lst,min(n,len(lst)))"),
    ]
    name,params,body = templates[np.random.randint(0,len(templates))]
    v = [f"def {name}({', '.join(params)}):\n    \"\"\"{name}\"\"\"\n    {body}",
         f"def {name}({', '.join(params)}):\n    #{name}\n    {body}"]
    return v[np.random.randint(0,len(v))]

def func_embed(text):
    vec = torch.zeros(EMBED_DIM)
    for i in range(len(text)-1):
        vec[hash(text[i:i+2])%EMBED_DIM] += 1.0
    return vec / (vec.norm()+1e-8)

class FuncBank:
    def __init__(self, n, nc=4): self.nc=nc; self.funcs=[gen_function() for _ in range(n)]; self.embs=torch.stack([func_embed(f) for f in self.funcs])
    def sample(self, bs):
        B,N=bs,self.nc; cand=torch.zeros(B,N,EMBED_DIM); tpos=torch.zeros(B,dtype=torch.long); txts=[]
        for b in range(B):
            tidx=np.random.randint(0,len(self.funcs)); txts.append(self.funcs[tidx])
            dist=[]; [dist.append(d) for d in np.random.choice([x for x in range(len(self.funcs)) if x!=tidx],N-1,replace=False)]
            all_idx=[tidx]+list(dist); np.random.shuffle(all_idx)
            for n in range(N): cand[b,n]=self.embs[all_idx[n]]
            tpos[b]=all_idx.index(tidx)
        return cand,tpos,txts

# ---- Models ----
class CNNSender(nn.Module):
    def __init__(self,ld): super().__init__(); self.ce=nn.Embedding(128,32); self.c1=nn.Conv1d(32,64,3,padding=1); self.c2=nn.Conv1d(64,64,5,padding=2); self.fc=nn.Sequential(nn.Linear(128,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,ld),nn.LayerNorm(ld)); self.vh=nn.Linear(HIDDEN,1)
    def forward(self,txts):
        B=len(txts);L=min(max(len(t) for t in txts),300);ids=torch.zeros(B,L,dtype=torch.long)
        for b,t in enumerate(txts):
            for i,ch in enumerate(t[:L]): ids[b,i]=min(ord(ch)%128,127)
        e=self.ce(ids).permute(0,2,1); c1=torch.relu(self.c1(e)); c2=torch.relu(self.c2(c1)); f=torch.cat([c1.mean(-1),c2.mean(-1)],-1); h=torch.relu(self.fc[0](f))
        return self.fc[2](self.fc[1](h)),self.vh(h).squeeze(-1)

class LSTMSender(nn.Module):
    def __init__(self,ld): super().__init__(); self.ce=nn.Embedding(128,32); self.lstm=nn.LSTM(32,64,2,bidirectional=True,batch_first=True); self.reduce=nn.Linear(256,HIDDEN); self.zh=nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,ld),nn.LayerNorm(ld)); self.vh=nn.Linear(HIDDEN,1)
    def forward(self,txts):
        B=len(txts);L=min(max(len(t) for t in txts),300);ids=torch.zeros(B,L,dtype=torch.long)
        for b,t in enumerate(txts):
            for i,ch in enumerate(t[:L]): ids[b,i]=min(ord(ch)%128,127)
        _,(hn,_)=self.lstm(self.ce(ids)); h=torch.relu(self.reduce(torch.cat([hn[i] for i in range(4)],-1)))
        return self.zh[2](self.zh[1](self.zh[0](h))),self.vh(h).squeeze(-1)

class AttnSender(nn.Module):
    def __init__(self,ld): super().__init__(); self.ce=nn.Embedding(128,32); self.pe=nn.Parameter(torch.randn(1,300,32)*0.02); self.attn=nn.MultiheadAttention(32,4,batch_first=True); self.zh=nn.Sequential(nn.Linear(32,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,ld),nn.LayerNorm(ld)); self.vh=nn.Linear(HIDDEN,1)
    def forward(self,txts):
        B=len(txts);L=min(max(len(t) for t in txts),300);ids=torch.zeros(B,L,dtype=torch.long)
        for b,t in enumerate(txts):
            for i,ch in enumerate(t[:L]): ids[b,i]=min(ord(ch)%128,127)
        e=self.ce(ids)+self.pe[:,:L,:]; a,_=self.attn(e,e,e); p=a.mean(1); h=torch.relu(self.zh[0](p))
        return self.zh[2](self.zh[1](h)),self.vh(h).squeeze(-1)

class Receiver(nn.Module):
    def __init__(self,ld): super().__init__(); self.cn=nn.Sequential(nn.Linear(EMBED_DIM,HIDDEN),nn.ReLU()); self.sc=nn.Sequential(nn.Linear(HIDDEN+ld,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,1))
    def forward(self,cand,z): B,N,_=cand.shape; h=self.cn(cand.view(B*N,EMBED_DIM)); ze=z.unsqueeze(1).expand(B,N,LATENT_DIM).reshape(B*N,LATENT_DIM); return self.sc(torch.cat([h,ze],-1)).view(B,N)

def train(sender,receiver,nc=4,epochs=4000):
    params=list(sender.parameters())+list(receiver.parameters()); opt=optim.Adam(params,lr=1e-3); sch=optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=0.0; bank=FuncBank(500,nc)
    for ep in range(epochs):
        cand,tpos,txts=bank.sample(32); z,v=sender(txts); logits=receiver(cand,z)
        dist=D.Categorical(logits=logits); acts=dist.sample(); lps=dist.log_prob(acts); r=(acts==tpos).float()*2-1
        loss=-(lps*(r-v.detach())).mean()+0.5*nn.MSELoss()(v,r)-0.02*dist.entropy().mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        acc=(acts==tpos).float().mean().item()
        if acc>best:best=acc
    return best

def evaluate(sender,receiver,nc=4,n=300):
    c=0;nc0=0;bank=FuncBank(200,nc)
    for _ in range(n):
        cand,tpos,txts=bank.sample(1)
        with torch.no_grad():
            z,_=sender(txts);pred=receiver(cand,z).argmax(-1).item();pred_n=receiver(cand,torch.randn(1,LATENT_DIM)*0.5).argmax(-1).item()
        if pred==tpos[0].item():c+=1
        if pred_n==tpos[0].item():nc0+=1
    return {"acc":c/n,"null":nc0/n,"over":c/n-1.0/nc}

# ---- Main ----
if __name__ == "__main__":
    pairs = [("CNN",CNNSender),("LSTM",LSTMSender),("Attn",AttnSender)]
    results = []
    for sname, SenderClass in pairs:
        print(f"{sname}...",end=" ",flush=True)
        s = SenderClass(LATENT_DIM); r = Receiver(LATENT_DIM)
        best = train(s,r); ev = evaluate(s,r)
        results.append({"pair":sname,"best":best,"test":ev["acc"],"null":ev["null"],"over":ev["over"]})
        print(f"best={best:.1%} test={ev['acc']:.1%} null={ev['null']:.1%} over={ev['over']:+.1%}",flush=True)

    print(f"\n{'Arch':>8s} {'Test':>8s} {'Null':>8s} {'Over':>8s} {'Verdict':>10s}")
    print("-"*50)
    for r in results:
        v="EMERGENT" if r["over"]>0.05 else "NO SIGNAL"
        print(f"{r['pair']:>8s} {r['test']:>7.1%} {r['null']:>7.1%} {r['over']:>7.1%} {v:>10s}")

    fig,ax=plt.subplots(figsize=(8,5))
    labels=[r["pair"] for r in results];x=np.arange(len(labels))
    ax.bar(x-0.2,[r["test"] for r in results],0.35,label="Test",color="blue",alpha=0.7)
    ax.bar(x+0.2,[r["null"] for r in results],0.35,label="Null",color="red",alpha=0.4)
    ax.axhline(y=0.25,color="gray",ls=":",alpha=0.5,label="Chance")
    ax.set_xticks(x);ax.set_xticklabels(labels,rotation=0,ha="center")
    ax.set_ylabel("Accuracy");ax.set_title(f"Architecture Comparison ({LATENT_DIM}D bottleneck)")
    ax.legend();ax.grid(True,alpha=0.3,axis="y");ax.set_ylim(0,1.1)
    plt.tight_layout();plt.savefig(out_dir/"xarch_results.png",dpi=150);plt.close()
    print(f"\nSaved: {out_dir}/xarch_results.png")
