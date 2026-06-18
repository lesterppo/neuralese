"""
Neuralese Path C — State Synchronization
Two agents see different partial views of a shared world.
Observer encodes its view into z. Navigator must reconstruct Observer's view from z alone.
Pure information throughput benchmark — no actions, no navigation.

Setup:
- World: 20D structured state (10 features × 2 agents)
- Agent A: sees features 0-9, encodes → 12D z
- Agent B: sees ONLY z, must reconstruct features 0-9
- Agent B also sees its own features 10-19 (not communicated)
- Metric: reconstruction MSE, per-feature correlation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIM = 12
HIDDEN = 128
STATE_DIM = 20            # Total world state
OBSERVER_DIM = 10         # Features Agent A sees and must communicate
NAVIGATOR_LOCAL_DIM = 10  # Features Agent B sees locally
EPOCHS = 6000
BATCH = 64
LR = 1e-3


def generate_state(batch_size):
    """Generate random world states with structured correlations.
    State = [agent_a_view(10), agent_b_local(10)]
    Features have internal correlations (not i.i.d.)."""
    # Base latent factors that create correlations
    latent = torch.randn(batch_size, 4)
    # Transform latent → correlated features
    W_a = torch.randn(4, OBSERVER_DIM) * 0.5
    W_b = torch.randn(4, NAVIGATOR_LOCAL_DIM) * 0.5
    a_view = (latent @ W_a + torch.randn(batch_size, OBSERVER_DIM) * 0.3)
    b_local = (latent @ W_b + torch.randn(batch_size, NAVIGATOR_LOCAL_DIM) * 0.3)
    # Normalize to [-1, 1]
    a_view = torch.tanh(a_view)
    b_local = torch.tanh(b_local)
    return torch.cat([a_view, b_local], dim=-1), a_view, b_local


class Observer(nn.Module):
    """Sees 10 features → 12D z."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVER_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )
    def forward(self, x): return self.net(x)


class Navigator(nn.Module):
    """12D z + 10 local features → reconstructed 10 Observer features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + NAVIGATOR_LOCAL_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, OBSERVER_DIM),
        )
    def forward(self, z, local): return self.net(torch.cat([z, local], dim=-1))


# --- MUTUAL INFORMATION ESTIMATOR ---
# Simple: if the Navigator can reconstruct Observer's features better than
# a baseline that only uses local info, the z channel carries information.

class Baseline(nn.Module):
    """Navigator without z — only uses local features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NAVIGATOR_LOCAL_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, OBSERVER_DIM),
        )
    def forward(self, local): return self.net(local)


def train(observer, navigator, baseline=None, epochs=EPOCHS):
    params = list(observer.parameters()) + list(navigator.parameters())
    opt = optim.Adam(params, lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"mse": [], "baseline_mse": [], "mi_gain": []}

    # Train baseline once on same data distribution
    if baseline:
        b_opt = optim.Adam(baseline.parameters(), lr=LR)
        for _ in range(1000):
            _, a_view, b_local = generate_state(BATCH)
            pred = baseline(b_local); loss = nn.MSELoss()(pred, a_view)
            b_opt.zero_grad(); loss.backward(); b_opt.step()

    for ep in range(epochs):
        _, a_view, b_local = generate_state(BATCH)
        z = observer(a_view)
        pred = navigator(z, b_local)
        loss = nn.MSELoss()(pred, a_view)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        hist["mse"].append(loss.item())

        if baseline:
            with torch.no_grad():
                b_pred = baseline(b_local)
                b_mse = nn.MSELoss()(b_pred, a_view).item()
                hist["baseline_mse"].append(b_mse)
                hist["mi_gain"].append(b_mse / max(loss.item(), 1e-8) - 1.0)

        if ep % 2000 == 0:
            print(f"  Step {ep:5d}: mse={loss.item():.6f}" +
                  (f" baseline={b_mse:.6f} mi_gain={hist['mi_gain'][-1]:.1%}" if baseline else ""))

    return hist


def evaluate_full(observer, navigator, baseline=None, n=500):
    """Comprehensive evaluation: per-feature MSE, correlation, MI gain."""
    _, a_view, b_local = generate_state(n)
    with torch.no_grad():
        z = observer(a_view)
        pred = navigator(z, b_local)

    mse_total = nn.MSELoss()(pred, a_view).item()
    # Per-feature correlation
    corrs = []
    for i in range(OBSERVER_DIM):
        c = np.corrcoef(pred[:, i].numpy(), a_view[:, i].numpy())[0, 1]
        corrs.append(c)
    avg_corr = np.mean(corrs)

    result = {"mse": mse_total, "avg_correlation": avg_corr, "per_feature_corr": corrs}

    if baseline:
        b_pred = baseline(b_local)
        b_mse = nn.MSELoss()(b_pred, a_view).item()
        mi_gain = b_mse / max(mse_total, 1e-8) - 1.0
        result["baseline_mse"] = b_mse
        result["mi_gain"] = mi_gain

    return result


def plot_results(hist, eval_r, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(hist["mse"], alpha=0.5, color='blue', label='Neuralese')
    if "baseline_mse" in hist:
        ax.plot(hist["baseline_mse"], alpha=0.3, color='red', label='No-z baseline')
        ax.legend()
    ax.set_title("Reconstruction MSE"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "mi_gain" in hist:
        ax.plot(hist["mi_gain"], alpha=0.5, color='green')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.set_title(f"MI Gain (how much z helps)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    corrs = eval_r["per_feature_corr"]
    colors = ['green' if c > 0.5 else 'orange' if c > 0.2 else 'red' for c in corrs]
    ax.bar(range(len(corrs)), corrs, color=colors, alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Strong')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.3, label='Weak')
    ax.set_title(f"Per-Feature Correlation (avg={eval_r['avg_correlation']:.3f})")
    ax.set_xlabel("Feature"); ax.set_ylabel("r"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    text = (f"MSE: {eval_r['mse']:.6f}\n"
            f"Avg Correlation: {eval_r['avg_correlation']:.3f}\n"
            f"Features: {OBSERVER_DIM}D → {LATENT_DIM}D → {OBSERVER_DIM}D\n"
            f"Bits: {LATENT_DIM*32} channel, {OBSERVER_DIM*32} state\n"
            f"Compression: {OBSERVER_DIM/LATENT_DIM:.1f}x")
    if "mi_gain" in eval_r:
        text += f"\nMI Gain: {eval_r['mi_gain']:.1%}"
    ax.text(0.1, 0.5, text, fontfamily='monospace', fontsize=10, va='center', transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle(f"Neuralese Path C — State Synchronization ({LATENT_DIM}D channel)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "path_c_results.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"NEURALESE PATH C — State Synchronization")
    print("=" * 60)
    print(f"  Agent A: sees {OBSERVER_DIM}D → encodes to {LATENT_DIM}D z")
    print(f"  Agent B: sees z + {NAVIGATOR_LOCAL_DIM}D local → reconstructs {OBSERVER_DIM}D")
    print(f"  Baseline: Agent B without z (local features only)")

    print("\n[Training]...")
    observer = Observer(); navigator = Navigator(); baseline = Baseline()
    hist = train(observer, navigator, baseline)

    print("\n[Evaluation]...")
    eval_r = evaluate_full(observer, navigator, baseline, n=500)
    print(f"  Neuralese MSE:     {eval_r['mse']:.6f}")
    print(f"  Baseline MSE (no z): {eval_r.get('baseline_mse', 0):.6f}")
    print(f"  MI Gain:            {eval_r.get('mi_gain', 0):.1%}")
    print(f"  Avg Correlation:    {eval_r['avg_correlation']:.3f}")
    print(f"  Strong features (r>0.5): {sum(1 for c in eval_r['per_feature_corr'] if c>0.5)}/{OBSERVER_DIM}")
    print(f"  Weak features (r>0.2):   {sum(1 for c in eval_r['per_feature_corr'] if c>0.2)}/{OBSERVER_DIM}")

    print("\nGenerating plots...")
    plot_results(hist, eval_r, out_dir)
    print(f"\nDone! Outputs in {out_dir}/")
