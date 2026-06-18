"""Neuralese Path B Stress Test — REAL compression benchmark.
Tests: open vocabulary (hash-based, cannot memorize) vs closed vocabulary (fixed set).
Key question: at what bottleneck dimension does compression break?
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, hashlib
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HIDDEN, EMBED_DIM, BATCH, LR = 128, 32, 64, 1e-3
LINE_MAX, NUM_EDITS = 2000, 6
EDITS = ["replace","insert","delete","append","prepend","modify"]
OBS_INPUT_DIM = 32 + 12 + 6 + 32 + 96 + 32

def hash_str(s, d=32):
    if not s: return torch.zeros(d)
    hh = hashlib.sha256(s.encode()).digest()
    bits = [float((b >> i) & 1) for b in hh for i in range(8)]
    vals = bits[:d] + [0.0] * max(0, d - len(bits))
    return torch.tensor(vals, dtype=torch.float32)

def gen_task(open_vocab=True):
    import uuid
    if open_vocab:
        m = np.random.choice(["src","lib","tests","tools","agent","utils"])
        f = f"{m}/{uuid.uuid4().hex[:8]}.py"
        verbs = ['get','set','process','handle','compute','load']
        fn = f"{np.random.choice(verbs)}{uuid.uuid4().hex[:6]}"
        errs = ['TypeError','ValueError','KeyError','RuntimeError']
        e = f"{np.random.choice(errs)}: {uuid.uuid4().hex[:10]}"
    else:
        f = np.random.choice([f"src/m{i}.py" for i in range(30)])
        fnames = ["evaluate","train","forward","backward","optimize",
                  "load","save","process","validate","transform","encode","decode"]
        fn = np.random.choice(fnames)
        e = np.random.choice(["TypeError: x","KeyError: k","ValueError: v",
                              "RuntimeError: r","IndexError: i"])
    ln = np.random.randint(1, LINE_MAX)
    ed = np.random.choice(EDITS)
    ctx = [f"def {fn}(a,b=None):",
           f"    r = self.{np.random.choice(['forward','apply'])}(a)",
           "    return data"]
    return f, ln, ed, fn, ctx, e

class TaskBank:
    def __init__(self, n, open_vocab=True):
        self.x = torch.zeros(n, OBS_INPUT_DIM)
        self.ft = torch.zeros(n, EMBED_DIM)
        self.lt = torch.zeros(n)
        self.et = torch.zeros(n, dtype=torch.long)
        self.fnt = torch.zeros(n, EMBED_DIM)
        for i in range(n):
            f, ln, ed, fn, ctx, err = gen_task(open_vocab)
            off = 0
            self.x[i, off:off+32] = hash_str(f); off += 32
            for b in range(12):
                if ln & (1 << b):
                    self.x[i, off+b] = 1.0
            off += 12
            self.x[i, off+EDITS.index(ed)] = 1.0; off += 6
            self.x[i, off:off+32] = hash_str(fn); off += 32
            for c in ctx:
                self.x[i, off:off+32] = hash_str(c); off += 32
            self.x[i, off:off+32] = hash_str(err)
            self.ft[i] = hash_str(f)
            self.lt[i] = ln / LINE_MAX
            self.et[i] = EDITS.index(ed)
            self.fnt[i] = hash_str(fn)

class Observer(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_INPUT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(),
            nn.Linear(HIDDEN//2, ld),
            nn.LayerNorm(ld))
    def forward(self, x):
        return self.net(x)

class Navigator(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(ld, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.fh = nn.Linear(HIDDEN, EMBED_DIM)
        self.lh = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())
        self.eh = nn.Linear(HIDDEN, NUM_EDITS)
        self.fnh = nn.Linear(HIDDEN, EMBED_DIM)
    def forward(self, z):
        h = self.shared(z)
        return self.fh(h), self.lh(h).squeeze(-1), self.eh(h), self.fnh(h)

def run_sweep(open_vocab, label, dims=None, epochs=1000):
    if dims is None:
        dims = [2, 4, 8, 16]
    bank = TaskBank(2000, open_vocab)
    results = []
    for ld in dims:
        print(f"  {label} {ld}D...", end=" ", flush=True)
        o = Observer(ld)
        n = Navigator(ld)
        opt = optim.Adam(list(o.parameters()) + list(n.parameters()), lr=LR)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            idx = np.random.choice(2000, BATCH, replace=False)
            x = bank.x[idx]
            ft = bank.ft[idx]
            lt = bank.lt[idx]
            et = bank.et[idx]
            fnt = bank.fnt[idx]
            pf, pl, pe, pfn = n(o(x))
            loss = (nn.MSELoss()(pf, ft) + nn.MSELoss()(pl, lt)
                    + nn.CrossEntropyLoss()(pe, et) + nn.MSELoss()(pfn, fnt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            pf, pl, pe, pfn = n(o(bank.x[:200]))
        ea = (pe.argmax(-1) == bank.et[:200]).float().mean().item()
        la = (pl - bank.lt[:200]).abs().mean().item() * LINE_MAX
        fc = nn.CosineSimilarity()(pf, bank.ft[:200]).mean().item()
        fnc = nn.CosineSimilarity()(pfn, bank.fnt[:200]).mean().item()
        r = {"dim": ld, "edit_acc": ea, "line_mae_abs": la,
             "file_cos": fc, "func_cos": fnc}
        results.append(r)
        print(f"edit={ea:.1%} line={la:.0f} fcos={fc:.3f}")
    return results

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    print("=" * 70)
    print("NEURALESE PATH B STRESS TEST")
    print("=" * 70)
    dims = [2, 4, 8, 16]
    print(f"  {OBS_INPUT_DIM}D input -> {dims}D bottleneck, {1000} epochs each")
    print()

    print("[1/2] Open vocabulary (hash-based, cannot memorize)...")
    open_r = run_sweep(True, "Open")
    print("\n[2/2] Closed vocabulary (fixed-set, can memorize)...")
    closed_r = run_sweep(False, "Closed")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ox, cy = open_r, closed_r
    dx = [r["dim"] for r in ox]

    axes[0, 0].plot(dx, [r["edit_acc"] for r in ox], 'g-o', label='Open vocab')
    axes[0, 0].plot(dx, [r["edit_acc"] for r in cy], 'g--s', label='Closed vocab')
    axes[0, 0].axhline(y=1/6, color='red', ls='--', alpha=0.3, label='Chance')
    axes[0, 0].set_title("Edit Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(dx, [r["line_mae_abs"] for r in ox], 'r-o', label='Open')
    axes[0, 1].plot(dx, [r["line_mae_abs"] for r in cy], 'r--s', label='Closed')
    axes[0, 1].set_title("Line MAE (absolute)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(dx, [r["file_cos"] for r in ox], 'b-o', label='File cos (open)')
    axes[1, 0].plot(dx, [r["func_cos"] for r in ox], 'orange', marker='o',
                    label='Func cos (open)')
    axes[1, 0].plot(dx, [r["file_cos"] for r in cy], 'b--s',
                    label='File cos (closed)')
    axes[1, 0].plot(dx, [r["func_cos"] for r in cy], 'orange', marker='s', ls='--',
                    label='Func cos (closed)')
    axes[1, 0].set_title("Cosine Similarity (open-vocab strings)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    open_fcos = [r["file_cos"] for r in ox]
    open_fncos = [r["func_cos"] for r in ox]
    closed_fcos = [r["file_cos"] for r in cy]
    closed_fncos = [r["func_cos"] for r in cy]
    summary_text = (
        f"OPEN VOCAB (hash):\n"
        f"File cos: {[f'{v:.3f}' for v in open_fcos]}\n"
        f"Func cos: {[f'{v:.3f}' for v in open_fncos]}\n\n"
        f"CLOSED VOCAB:\n"
        f"File cos: {[f'{v:.3f}' for v in closed_fcos]}\n"
        f"Func cos: {[f'{v:.3f}' for v in closed_fncos]}")
    axes[1, 1].text(0.1, 0.5, summary_text,
                    fontfamily='monospace', fontsize=8, va='center',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Summary")
    axes[1, 1].axis('off')

    plt.suptitle("Neuralese Path B Stress — Open vs Closed Vocabulary",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "path_b_stress_results.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir}/path_b_stress_results.png")

    print("\n" + "=" * 70)
    print(f"{'Dim':>5} {'Open Edit':>12} {'Open Line':>11} {'Open FCos':>11} "
          f"{'Closed Edit':>14} {'Closed Line':>13} {'Closed FCos':>13}")
    print("-" * 70)
    for o, c in zip(open_r, closed_r):
        print(f"{o['dim']:>5} {o['edit_acc']:>11.1%} {o['line_mae_abs']:>10.0f} "
              f"{o['file_cos']:>10.3f} {c['edit_acc']:>13.1%} "
              f"{c['line_mae_abs']:>12.0f} {c['file_cos']:>12.3f}")
