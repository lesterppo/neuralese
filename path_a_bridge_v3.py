"""
Neuralese Path A Bridge v3 — Hermes Subagent Context Compression
Uses Path B's proven architecture: learnable embeddings + supervised training.
Gemini Session 5 Round 2: weighted loss for categorical fields, not larger bottleneck.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIM = 16
HIDDEN = 128
EMBED_DIM = 32
EPOCHS = 8000
BATCH = 64
LR = 1e-3
CATEGORICAL_WEIGHT = 10.0  # Weight for one-hot fields (Gemini R2 recommendation)

# Vocabulary
MAX_FILES = 50
MAX_ERROR_TYPES = 32
MAX_FUNCTIONS = 128
MAX_FIXES = 3
FIX_VOCAB = 60
MAX_SYMBOLS = 5
SYMBOL_VOCAB = 100
MAX_WORKSPACES = 20

INPUT_DIM = (EMBED_DIM + 12 + MAX_ERROR_TYPES + MAX_FUNCTIONS
             + MAX_FIXES * EMBED_DIM + MAX_SYMBOLS * EMBED_DIM + EMBED_DIM)
# 32+12+32+128+96+160+32 = 492


@dataclass
class AgentContext:
    file_path: str; line_number: int; error_type: str
    function_name: str; attempted_fixes: List[str]; related_symbols: List[str]
    workspace_path: str


ERRORS = ["type_error","import_error","value_error","attribute_error",
    "key_error","index_error","name_error","syntax_error","runtime_error",
    "os_error","io_error","assertion_error","timeout_error","memory_error",
    "connection_error","permission_error","not_implemented","config_error",
    "auth_error","api_error","data_error","logic_error","race_condition",
    "null_reference","overflow_error","underflow_error","encoding_error",
    "serialization_error","integration_error","deprecation_warning",
    "resource_exhausted","deadlock"]

FUNCTIONS = (list(dict.fromkeys([
    "evaluate","train","warm_start","rl_fine_tune","ppo_update",
    "collect_trajectories","compute_gae","gen_maze","bfs_reachable","a_star_path",
    "forward","backward","step","optimize","load_config","save_model",
    "read_file","write_file","search_files","patch","terminal","execute_code",
    "delegate_task","handle_function_call","chat","run_conversation",
    "build_prompt","parse_response","generate","classify","encode","decode",
    "compress","decompress","validate","transform","preprocess","postprocess",
    "normalize","extract_features","reduce_dim","augment","process_batch",
    "handle_error","format_output","sanitize_input","log_metrics",
    "apply_patch","compute_hash","verify_signature","serialize","deserialize",
    "detect_anomaly","filter_noise","interpolate","convolve","quantize",
    "threshold","smooth","align_sequences","cluster_data","cross_validate",
    "compute_gradient","apply_dropout","batch_norm","layer_norm","attention",
])) * 3)[:MAX_FUNCTIONS]

FILES = [f"src/module_{i}.py" for i in range(20)] + [
    f"lib/utils_{i}.py" for i in range(10)] + [
    f"tests/test_{i}.py" for i in range(10)] + [
    f"tools/{t}.py" for t in ["delegate","terminal","file","search"]] + [
    f"agent/{a}.py" for a in ["memory","cache","compression","display","curator"]]

WORKSPACES = ["/home/peter/neuralese","/home/peter/hermes-agent","/home/peter/deepworld",
              "/home/peter/ply-tensor-language","/home/peter/stock-world-model",
              "/home/peter/doctor-roster","/home/peter/research","/home/peter/gemini-cli"]

FIXES = [f"added {op} to fix {what}" for op in
    [".squeeze(0)",".unsqueeze(0)",".detach()",".clamp()","torch.no_grad()",
     "try/except","null check","bounds check","type cast","assert"] for what in
    ["tensor shape","NoneType","index error","type mismatch","gradient flow"]][:FIX_VOCAB]

SYMBOLS = ["navigator","observer","evaluate","forward","delegate_task","chat",
    "run_conversation","load_config","save_output","validate_input","ppo_update",
    "compute_gae","warm_start","collect_trajectories","handle_error","format_output",
    "apply_patch","compute_hash","verify_signature","serialize","deserialize",
    "detect_anomaly","filter_noise","interpolate","convolve","quantize",
    "threshold","smooth","align_sequences","cluster_data","cross_validate",
    "compute_gradient","apply_dropout","batch_norm","layer_norm","attention",
    "embed","tokenize","parse_args","build_prompt","parse_response","generate",
    "classify","encode","decode","compress","decompress","validate","transform",
    "preprocess","postprocess","normalize","extract_features","reduce_dim",
    "augment","process_batch","log_metrics","shuffle","split_dataset","merge_configs",
    "sanitize_input","cache_result","check_bounds","rank_items","predict",
    "rank","sort","filter","map","reduce","join","split","strip","replace",
    "find","search","match","sub","gsub","format","sprintf","interpolate",
    "interleave","zip","unzip","group_by","partition","flatten","reshape"][:SYMBOL_VOCAB]


class TaskBank:
    def __init__(self, size=5000):
        self.tasks = [self._gen() for _ in range(size)]

    def _gen(self):
        return AgentContext(
            file_path=np.random.choice(FILES),
            line_number=np.random.randint(1, 2000),
            error_type=np.random.choice(ERRORS),
            function_name=np.random.choice(FUNCTIONS),
            attempted_fixes=list(np.random.choice(FIXES,
                size=np.random.randint(0, min(4, len(FIXES))), replace=False)),
            related_symbols=list(np.random.choice(SYMBOLS,
                size=np.random.randint(1, min(6, len(SYMBOLS))), replace=False)),
            workspace_path=np.random.choice(WORKSPACES),
        )

    def sample(self, n):
        return [self.tasks[i] for i in np.random.choice(len(self.tasks), n, replace=False)]

    def to_input(self, tasks, file_emb, fix_emb, sym_emb, ws_emb):
        B = len(tasks); x = torch.zeros(B, INPUT_DIM)
        for b, t in enumerate(tasks):
            off = 0
            x[b,off:off+EMBED_DIM]=file_emb(torch.tensor(FILES.index(t.file_path))); off+=EMBED_DIM
            for i in range(12):
                if t.line_number & (1<<i): x[b,off+i]=1.0
            off+=12
            x[b,off+ERRORS.index(t.error_type)]=1.0; off+=MAX_ERROR_TYPES
            x[b,off+FUNCTIONS.index(t.function_name)]=1.0; off+=MAX_FUNCTIONS
            for i,fix in enumerate(t.attempted_fixes[:MAX_FIXES]):
                fid=FIXES.index(fix) if fix in FIXES else 0
                x[b,off+i*EMBED_DIM:off+(i+1)*EMBED_DIM]=fix_emb(torch.tensor(fid))
            off+=MAX_FIXES*EMBED_DIM
            for i,sym in enumerate(t.related_symbols[:MAX_SYMBOLS]):
                sid=SYMBOLS.index(sym) if sym in SYMBOLS else 0
                x[b,off+i*EMBED_DIM:off+(i+1)*EMBED_DIM]=sym_emb(torch.tensor(sid))
            off+=MAX_SYMBOLS*EMBED_DIM
            x[b,off:off+EMBED_DIM]=ws_emb(torch.tensor(WORKSPACES.index(t.workspace_path)))
        return x

    def targets(self, tasks):
        B=len(tasks)
        files=torch.zeros(B,dtype=torch.long); lines=torch.zeros(B)
        errs=torch.zeros(B,dtype=torch.long); fns=torch.zeros(B,dtype=torch.long)
        for b,t in enumerate(tasks):
            files[b]=FILES.index(t.file_path); lines[b]=t.line_number/2000.0
            errs[b]=ERRORS.index(t.error_type); fns[b]=FUNCTIONS.index(t.function_name)
        return files,lines,errs,fns


class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),
            nn.Linear(HIDDEN,HIDDEN//2),nn.ReLU(),nn.Linear(HIDDEN//2,LATENT_DIM),
            nn.LayerNorm(LATENT_DIM))
    def forward(self,x): return self.net(x)


class Navigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(LATENT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.file_head=nn.Linear(HIDDEN,MAX_FILES)
        self.line_head=nn.Sequential(nn.Linear(HIDDEN,1),nn.Sigmoid())
        self.err_head=nn.Linear(HIDDEN,MAX_ERROR_TYPES)
        self.fn_head=nn.Linear(HIDDEN,MAX_FUNCTIONS)
    def forward(self,z):
        h=self.shared(z)
        return self.file_head(h),self.line_head(h).squeeze(-1),self.err_head(h),self.fn_head(h)


def train(observer,navigator,bank,file_emb,fix_emb,sym_emb,ws_emb,epochs=EPOCHS):
    params=(list(observer.parameters())+list(navigator.parameters())+
            list(file_emb.parameters())+list(fix_emb.parameters())+
            list(sym_emb.parameters())+list(ws_emb.parameters()))
    opt=optim.Adam(params,lr=LR); sch=optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    hist={"loss":[],"file_acc":[],"line_mae":[],"err_acc":[],"fn_acc":[]}
    for ep in range(epochs):
        tasks=bank.sample(BATCH)
        x=bank.to_input(tasks,file_emb,fix_emb,sym_emb,ws_emb)
        tf,tl,te,tfn=bank.targets(tasks)
        z=observer(x); pf,pl,pe,pfn=navigator(z)
        lf=nn.CrossEntropyLoss()(pf,tf)
        ll=nn.MSELoss()(pl,tl)
        le=nn.CrossEntropyLoss()(pe,te)*CATEGORICAL_WEIGHT  # Weighted!
        lfn=nn.CrossEntropyLoss()(pfn,tfn)*CATEGORICAL_WEIGHT  # Weighted!
        loss=lf+ll+le+lfn
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        hist["loss"].append(loss.item())
        hist["file_acc"].append((pf.argmax(-1)==tf).float().mean().item())
        hist["line_mae"].append((pl-tl).abs().mean().item())
        hist["err_acc"].append((pe.argmax(-1)==te).float().mean().item())
        hist["fn_acc"].append((pfn.argmax(-1)==tfn).float().mean().item())
        if ep%2000==0:
            print(f"  Step {ep:5d}: loss={loss.item():.4f} file={hist['file_acc'][-1]:.1%} "
                  f"err={hist['err_acc'][-1]:.1%} fn={hist['fn_acc'][-1]:.1%} line={hist['line_mae'][-1]:.3f}")
    return hist


def evaluate(observer,navigator,bank,file_emb,fix_emb,sym_emb,ws_emb,n=200):
    tasks=bank.sample(n)
    x=bank.to_input(tasks,file_emb,fix_emb,sym_emb,ws_emb)
    tf,tl,te,tfn=bank.targets(tasks)
    with torch.no_grad():
        pf,pl,pe,pfn=navigator(observer(x))
    fa=(pf.argmax(-1)==tf).float().mean().item()
    lm=(pl-tl).abs().mean().item()
    ea=(pe.argmax(-1)==te).float().mean().item()
    fna=(pfn.argmax(-1)==tfn).float().mean().item()
    em=((pf.argmax(-1)==tf)&(pe.argmax(-1)==te)&(pfn.argmax(-1)==tfn)).float().mean().item()
    return {"file_acc":fa,"line_mae":lm,"line_mae_abs":lm*2000,"err_acc":ea,"fn_acc":fna,"exact_match":em}


def plot_results(hist,er,out_dir):
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    axes[0,0].plot(hist["loss"],alpha=0.3,color='blue'); axes[0,0].set_title("Loss"); axes[0,0].set_yscale("log"); axes[0,0].grid(True,alpha=0.3)
    axes[0,1].plot(hist["file_acc"],alpha=0.6,label="File",color='green')
    axes[0,1].plot(hist["err_acc"],alpha=0.6,label="Error",color='orange')
    axes[0,1].plot(hist["fn_acc"],alpha=0.6,label="Function",color='blue')
    axes[0,1].set_title("Categorical Accuracy"); axes[0,1].legend(); axes[0,1].set_ylim(0,1.05); axes[0,1].grid(True,alpha=0.3)
    axes[1,0].plot(hist["line_mae"],alpha=0.6,color='red'); axes[1,0].set_title("Line MAE (norm)"); axes[1,0].grid(True,alpha=0.3)
    ms=["File","Error","Function","Exact"]
    vs=[er["file_acc"],er["err_acc"],er["fn_acc"],er["exact_match"]]
    bars=axes[1,1].bar(ms,vs,color=['green','orange','blue','purple'],alpha=0.7)
    for b,v in zip(bars,vs): axes[1,1].text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{v:.1%}",ha='center')
    axes[1,1].set_title("Evaluation"); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(True,alpha=0.3,axis='y')
    plt.suptitle(f"Neuralese Bridge v3 — {LATENT_DIM}D Context Compression",fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/"bridge_v3_results.png",dpi=150); plt.close()


if __name__=="__main__":
    out_dir=Path(__file__).parent/"output"; out_dir.mkdir(exist_ok=True)
    print("="*60)
    print(f"NEURALESE BRIDGE v3 — {LATENT_DIM}D Context Compression")
    print("="*60)
    print(f"  Categorical weight: {CATEGORICAL_WEIGHT}x (Gemini R2)")
    print(f"  Task bank: 5000 synthetic contexts")
    file_emb=nn.Embedding(MAX_FILES,EMBED_DIM); fix_emb=nn.Embedding(FIX_VOCAB,EMBED_DIM)
    sym_emb=nn.Embedding(SYMBOL_VOCAB,EMBED_DIM); ws_emb=nn.Embedding(MAX_WORKSPACES,EMBED_DIM)
    bank=TaskBank(5000)
    print("\n[Training]..."); observer=Observer(); navigator=Navigator()
    hist=train(observer,navigator,bank,file_emb,fix_emb,sym_emb,ws_emb)
    print("\n[Evaluation]..."); er=evaluate(observer,navigator,bank,file_emb,fix_emb,sym_emb,ws_emb)
    print(f"  File: {er['file_acc']:.1%}  Error: {er['err_acc']:.1%}  Function: {er['fn_acc']:.1%}")
    print(f"  Line MAE: {er['line_mae_abs']:.0f} lines  Exact match: {er['exact_match']:.1%}")
    bits=LATENT_DIM*32; text_bits=200*16.6
    print(f"  Bandwidth: {bits} bits vs ~{text_bits:.0f} bits (text) = {text_bits/bits:.0f}x")
    plot_results(hist,er,out_dir)
    print(f"\nDone! {out_dir}/")
