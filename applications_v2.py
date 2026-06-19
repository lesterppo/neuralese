"""Neuralese Applications v2 — Realistic Scale"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
LATENT_DIM, HIDDEN = 16, 128

TASK_TYPES = ["fix_bug","add_feature","refactor","write_test","review_code","optimize","document","debug"]
N_TASKS, N_FILES, N_FUNCS, N_LINES = 8, 50, 32, 2000

FILES = [f"{d}/m{i}.py" for d in ["src","lib","tools","agent","api","tests","utils","cli","models","handlers"] for i in range(5)][:N_FILES]
FUNCS = ["process","handle","validate","transform","load","save","compute","fetch","parse","format",
         "execute","delegate","search","patch","merge","filter","sort","cache","log","route",
         "serialize","deserialize","normalize","optimize","train","evaluate","forward","backward",
         "encode","decode","compress","decompress"][:N_FUNCS]
BUGS = ["TypeError: expected str got int","KeyError: key not found","IndexError: list index",
        "AttributeError: NoneType","RuntimeError: shape mismatch","ValueError: invalid literal",
        "FileNotFoundError","ImportError","TimeoutError","ConnectionError"]

def gen_task():
    bug = np.random.choice(BUGS); tt = np.random.choice(TASK_TYPES)
    file = np.random.choice(FILES); line = np.random.randint(1, N_LINES)
    func = np.random.choice(FUNCS); n_ctx = np.random.randint(2, 8)
    ctx_lines = "\n".join([f"  {i+1}: {np.random.choice(['def','class','if','for','return','import','try'])} ..." for i in range(n_ctx)])
    full = (f"SUBAGENT TASK #{np.random.randint(1000,9999)}\nType: {tt}\nFile: {file}:{line}\n"
            f"Function: {func}()\nError: {bug}\nContext ({n_ctx} lines):\n{ctx_lines}\n"
            f"Workspace: /workspace/project\nInstructions: Analyze and fix.")
    n_tokens = len(full.split()) * 1.3
    return {"task_type":tt,"file":file,"line":line,"func":func,"bug":bug,"full_text":full,"tokens":n_tokens}

def text_emb(text):
    v = torch.zeros(128)
    for i in range(len(text)-1): v[hash(text[i:i+2])%128] += 1.0
    return v / (v.norm() + 1e-8)

class Sender(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT_DIM), nn.LayerNorm(LATENT_DIM))

    def forward(self, emb):
        return self.net(emb)

class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.sh = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.task_h = nn.Linear(HIDDEN, N_TASKS)
        self.file_h = nn.Linear(HIDDEN, N_FILES)
        self.func_h = nn.Linear(HIDDEN, N_FUNCS)
        self.line_h = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())

    def forward(self, z):
        h = self.sh(z)
        return self.task_h(h), self.file_h(h), self.line_h(h).squeeze(-1), self.func_h(h)

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_TASKS))

    def forward(self, z):
        return self.net(z)

# ---- MAIN ----
if __name__ == "__main__":
    print("=" * 70)
    print("NEURALESE APPLICATIONS v2 — Realistic Scale")
    print("=" * 70)
    print(f"  Files: {N_FILES}, Functions: {N_FUNCS}, Tasks: {N_TASKS}")
    print(f"  Bottleneck: {LATENT_DIM}D ({LATENT_DIM*32} bits)")
    print()

    ntrain, ntest = 3000, 500
    train_tasks = [gen_task() for _ in range(ntrain)]
    test_tasks = [gen_task() for _ in range(ntest)]
    avg_tokens = np.mean([t["tokens"] for t in train_tasks])
    print(f"  Avg context: {avg_tokens:.0f} tokens (realistic delegate_task size)")
    print()

    # Train Sender+Receiver
    print("[1/2] Training context compressor...")
    s, r = Sender(), Receiver()
    params = list(s.parameters()) + list(r.parameters())
    opt = optim.Adam(params, lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 5000)

    for ep in range(5000):
        idx = np.random.choice(ntrain, 32)
        bt = [train_tasks[i] for i in idx]
        embs = torch.stack([text_emb(t["full_text"]) for t in bt])
        z = s(embs)
        pt, pf, pl, pfn = r(z)
        tt = torch.tensor([TASK_TYPES.index(t["task_type"]) for t in bt])
        tf = torch.tensor([FILES.index(t["file"]) for t in bt])
        tfn = torch.tensor([FUNCS.index(t["func"]) for t in bt])
        tl = torch.tensor([t["line"] / N_LINES for t in bt])
        loss = (nn.CrossEntropyLoss()(pt, tt) + nn.CrossEntropyLoss()(pf, tf) +
                nn.MSELoss()(pl, tl) + nn.CrossEntropyLoss()(pfn, tfn))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 1000 == 0:
            ta = (pt.argmax(-1) == tt).float().mean().item()
            fa = (pf.argmax(-1) == tf).float().mean().item()
            print(f"  ep{ep:5d}: task={ta:.1%} file={fa:.1%} loss={loss.item():.4f}", flush=True)

    # Evaluate
    print("\n[Evaluation]...")
    bt = [test_tasks[i] for i in range(ntest)]
    embs = torch.stack([text_emb(t["full_text"]) for t in bt])
    with torch.no_grad():
        z = s(embs)
        pt, pf, pl, pfn = r(z)
    tt = torch.tensor([TASK_TYPES.index(t["task_type"]) for t in bt])
    tf = torch.tensor([FILES.index(t["file"]) for t in bt])
    tfn = torch.tensor([FUNCS.index(t["func"]) for t in bt])
    ta = (pt.argmax(-1) == tt).float().mean().item()
    fa = (pf.argmax(-1) == tf).float().mean().item()
    fna = (pfn.argmax(-1) == tfn).float().mean().item()
    la = (pl.squeeze(-1) - torch.tensor([t["line"]/N_LINES for t in bt])).abs().mean().item() * N_LINES
    exact = ((pt.argmax(-1) == tt) & (pf.argmax(-1) == tf) & (pfn.argmax(-1) == tfn)).float().mean().item()
    vec_tokens = LATENT_DIM * 4 / 4

    print(f"  Task type:    {ta:.1%}")
    print(f"  File path:    {fa:.1%}  (chance={1/N_FILES:.1%})")
    print(f"  Function:     {fna:.1%}  (chance={1/N_FUNCS:.1%})")
    print(f"  Line MAE:     {la:.0f} lines")
    print(f"  Exact match:  {exact:.1%}")
    print(f"  Token savings: {avg_tokens:.0f} to {vec_tokens:.0f} = {avg_tokens/vec_tokens:.1f}x")

    # Train Router
    print(f"\n[2/2] Training task router on 16D vectors...")
    router = Router()
    ropt = optim.Adam(router.parameters(), lr=1e-3)
    rsch = optim.lr_scheduler.CosineAnnealingLR(ropt, 2000)
    for ep in range(2000):
        idx = np.random.choice(ntrain, 32)
        bt = [train_tasks[i] for i in idx]
        embs = torch.stack([text_emb(t["full_text"]) for t in bt])
        with torch.no_grad():
            z = s(embs)
        logits = router(z)
        targets = torch.tensor([TASK_TYPES.index(t["task_type"]) for t in bt])
        loss = nn.CrossEntropyLoss()(logits, targets)
        ropt.zero_grad(); loss.backward(); ropt.step(); rsch.step()

    # Router eval — use test_tasks, not bt (which was overwritten)
    test_embs = torch.stack([text_emb(t["full_text"]) for t in test_tasks])
    test_tt = torch.tensor([TASK_TYPES.index(t["task_type"]) for t in test_tasks])
    with torch.no_grad():
        tz = s(test_embs)
        tpreds = router(tz).argmax(-1)
    rfa = (tpreds == test_tt).float().mean().item()
    print(f"  Routing accuracy: {rfa:.1%} (chance={1/N_TASKS:.1%})")

    # Summary
    print(f"\n{'='*70}")
    print(f"APPLICATIONS v2 SUMMARY")
    print(f"{'='*70}")
    print(f"  1. CONTEXT COMPRESSION ({LATENT_DIM}D):")
    print(f"     Task: {ta:.1%} | File: {fa:.1%} | Func: {fna:.1%} | Line MAE: {la:.0f}")
    print(f"     Exact match: {exact:.1%} | Token savings: {avg_tokens/vec_tokens:.1f}x")
    verdict1 = "DEPLOYABLE" if exact > 0.3 else "PARTIAL — coarse fields work, fine fields need more dims"
    print(f"     Verdict: {verdict1}")
    print(f"  2. TASK ROUTING: {rfa:.1%} (chance={1/N_TASKS:.1%})")
    print(f"     Verdict: {'DEPLOYABLE' if rfa > 0.9 else 'NEEDS WORK'}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(["Task","File","Func","Exact"], [ta, fa, fna, exact],
                   color=["blue","green","orange","purple"], alpha=0.7)
    for i, v in enumerate([ta, fa, fna, exact]):
        axes[0, 0].text(i, v+0.01, f"{v:.1%}", ha="center")
    axes[0, 0].set_title(f"Context Reconstruction ({LATENT_DIM}D)")
    axes[0, 0].set_ylim(0, 1.1); axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(["Full text", "Neuralese"], [avg_tokens, vec_tokens],
                   color=["red", "green"], alpha=0.7)
    axes[0, 1].set_title(f"Token Savings: {avg_tokens/vec_tokens:.1f}x")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([avg_tokens, vec_tokens]):
        axes[0, 1].text(i, v+3, f"{v:.0f}", ha="center")

    axes[1, 0].bar(["Router (16D)", "Chance"], [rfa, 1/N_TASKS],
                   color=["purple", "gray"], alpha=0.7)
    axes[1, 0].set_title(f"Task Routing: {rfa:.1%}")
    axes[1, 0].set_ylim(0, 1.1); axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].text(0.1, 0.5,
        f"DEPLOYMENT READINESS\n{'='*22}\n\n"
        f"Task Routing: {'READY' if rfa>0.9 else 'NEEDS WORK'}\n"
        f"  {rfa:.1%} from 16D, zero text\n\n"
        f"Context Compression:\n"
        f"  {'VIABLE' if exact>0.3 else 'PARTIAL'}\n"
        f"  Task+file: usable\n"
        f"  Fine fields: need work\n\n"
        f"Token savings: {avg_tokens/vec_tokens:.1f}x\n"
        f"  {avg_tokens:.0f}t to {vec_tokens:.0f}t",
        fontfamily='monospace', fontsize=9, va='center',
        transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    plt.suptitle("Neuralese Applications v2 — Realistic Scale", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "applications_v2.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir}/applications_v2.png")
