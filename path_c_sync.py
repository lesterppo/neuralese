"""
Neuralese Path C — State Synchronization (FIXED)
Independent analysis fix: removes shared-latent confound.

OLD (tautological): Both agents' 10D states were linear projections of same 4D latent.
  → Navigator's local state + 12D z trivially reconstructs Observer's state because
    both views are deterministic functions of the SAME 4 latent factors.
  → "116,143% MI Gain" was a measure of how correlated the projection matrices were,
    not how much information the z channel carries.

NEW (fair): Agents see INDEPENDENTLY generated 10D feature vectors.
  OVERLAP_FEATURES: N features that are IDENTICAL between both agents (shared ground truth).
  PRIVATE_FEATURES: N features that are drawn INDEPENDENTLY per agent (no correlation).
  → Baseline (local info only) can guess shared features from Navigator's PRIVATE state
    because the shared features are present on BOTH sides.
  → Neuralese channel z must encode the Observer's unique private features, which the
    Navigator CANNOT infer from its own state.
  → MI Gain now measures genuine information throughput, not matrix similarity.

Controllable: OVERLAP_FEATURES parameter sweeps from 0 (fully independent, hardest)
  to OBSERVER_DIM (fully shared, easiest = old tautological setup).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIM = 8          # Bottleneck: TIGHTER than source (8D < 10D = real compression)
HIDDEN = 128
OBSERVER_DIM = 10          # Features Agent A sees
NAVIGATOR_DIM = 10          # Features Agent B sees locally
OVERLAP_FEATURES = 3        # Number of SHARED features (both agents see same values)
PRIVATE_FEATURES = 7        # Number of INDEPENDENT features per agent (different values)
EPOCHS = 6000
BATCH = 64
LR = 1e-3


def generate_state(batch_size, overlap=None, private=None):
    """
    Generate independent agent states with controlled overlap.

    Shared features (0..overlap): IDENTICAL values for both agents.
    Private features (overlap..OBSERVER_DIM): independently drawn per agent.

    If neither overlap nor private is passed, defaults to OVERLAP_FEATURES / PRIVATE_FEATURES.
    If only overlap is passed, private = OBSERVER_DIM - overlap.
    """
    if overlap is None:
        overlap = OVERLAP_FEATURES
    if private is None:
        private = OBSERVER_DIM - overlap
    parts_obs = []
    parts_nav = []
    shared = None

    if overlap > 0:
        shared = torch.randn(batch_size, overlap) * 0.8
        parts_obs.append(shared)
        parts_nav.append(shared)

    if private > 0:
        obs_private = torch.randn(batch_size, private) * 0.8
        nav_private = torch.randn(batch_size, private) * 0.8
        parts_obs.append(obs_private)
        parts_nav.append(nav_private)

    obs_state = torch.cat(parts_obs, dim=-1)        # [B, OBSERVER_DIM]
    nav_state = torch.cat(parts_nav, dim=-1)        # [B, NAVIGATOR_DIM]

    obs_state = torch.tanh(obs_state)
    nav_state = torch.tanh(nav_state)

    return obs_state, nav_state, shared if overlap > 0 else torch.zeros(batch_size, 0)


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
            nn.Linear(LATENT_DIM + NAVIGATOR_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, OBSERVER_DIM),
        )
    def forward(self, z, local):
        return self.net(torch.cat([z, local], dim=-1))


class Baseline(nn.Module):
    """Navigator without z — uses ONLY local features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NAVIGATOR_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, OBSERVER_DIM),
        )
    def forward(self, local):
        return self.net(local)


class Oracle(nn.Module):
    """Oracle baseline: Navigator that sees BOTH local state AND shared features directly.
    This is the UPPER BOUND — if Neuralese beats this, something is wrong."""
    def __init__(self, overlap=OVERLAP_FEATURES):
        super().__init__()
        self.overlap = overlap
        self.net = nn.Sequential(
            nn.Linear(NAVIGATOR_DIM + overlap, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, OBSERVER_DIM),
        )
    def forward(self, local, shared):
        if self.overlap == 0 or shared.shape[1] == 0:
            return self.net(local)
        return self.net(torch.cat([local, shared], dim=-1))


def train(observer, navigator, baseline=None, oracle=None, epochs=EPOCHS, overlap=OVERLAP_FEATURES):
    params = list(observer.parameters()) + list(navigator.parameters())
    opt = optim.Adam(params, lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"mse": [], "baseline_mse": [], "oracle_mse": [], "mi_gain": []}

    # Train baseline
    if baseline:
        b_opt = optim.Adam(baseline.parameters(), lr=LR)
        for _ in range(1000):
            obs_s, nav_s, _ = generate_state(BATCH, overlap=overlap)
            pred = baseline(nav_s)
            loss = nn.MSELoss()(pred, obs_s)
            b_opt.zero_grad(); loss.backward(); b_opt.step()

    # Train oracle
    if oracle:
        o_opt = optim.Adam(oracle.parameters(), lr=LR)
        for _ in range(1000):
            obs_s, nav_s, shared = generate_state(BATCH, overlap=overlap)
            pred = oracle(nav_s, shared)
            loss = nn.MSELoss()(pred, obs_s)
            o_opt.zero_grad(); loss.backward(); o_opt.step()

    for ep in range(epochs):
        obs_s, nav_s, shared = generate_state(BATCH, overlap=overlap)
        z = observer(obs_s)
        pred = navigator(z, nav_s)
        loss = nn.MSELoss()(pred, obs_s)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        hist["mse"].append(loss.item())

        if baseline:
            with torch.no_grad():
                b_mse = nn.MSELoss()(baseline(nav_s), obs_s).item()
                hist["baseline_mse"].append(b_mse)
                hist["mi_gain"].append(b_mse / max(loss.item(), 1e-8) - 1.0)

        if oracle:
            with torch.no_grad():
                o_mse = nn.MSELoss()(oracle(nav_s, shared), obs_s).item()
                hist["oracle_mse"].append(o_mse)

        if ep % 2000 == 0:
            parts = [f"Step {ep:5d}: mse={loss.item():.6f}"]
            if baseline:
                parts.append(f"baseline={hist['baseline_mse'][-1]:.6f} mi_gain={hist['mi_gain'][-1]:.1%}")
            if oracle:
                parts.append(f"oracle={hist['oracle_mse'][-1]:.6f}")
            print("  " + " ".join(parts))

    return hist


def evaluate_full(observer, navigator, baseline=None, oracle=None, n=500, overlap=OVERLAP_FEATURES):
    obs_s, nav_s, shared = generate_state(n, overlap=overlap)
    with torch.no_grad():
        z = observer(obs_s)
        pred = navigator(z, nav_s)

    mse_total = nn.MSELoss()(pred, obs_s).item()
    corrs = []
    for i in range(OBSERVER_DIM):
        c = np.corrcoef(pred[:, i].numpy(), obs_s[:, i].numpy())[0, 1]
        corrs.append(c)
    avg_corr = np.mean(corrs)

    result = {"mse": mse_total, "avg_correlation": avg_corr, "per_feature_corr": corrs}

    if baseline:
        b_pred = baseline(nav_s)
        b_mse = nn.MSELoss()(b_pred, obs_s).item()
        mi_gain = b_mse / max(mse_total, 1e-8) - 1.0
        result["baseline_mse"] = b_mse
        result["mi_gain"] = mi_gain

        # Per-feature breakdown: shared vs private
        shared_corrs = corrs[:overlap] if overlap > 0 else []
        private_corrs = corrs[overlap:] if overlap < OBSERVER_DIM else []
        result["shared_avg_corr"] = np.mean(shared_corrs) if shared_corrs else 0
        result["private_avg_corr"] = np.mean(private_corrs) if private_corrs else 0

    if oracle:
        o_pred = oracle(nav_s, shared)
        o_mse = nn.MSELoss()(o_pred, obs_s).item()
        result["oracle_mse"] = o_mse
        # Neuralese/Oracle ratio: close to 1.0 = near-optimal (good)
        result["neuralese_vs_oracle"] = mse_total / max(o_mse, 1e-8)

    return result


def sweep_overlap(out_dir):
    """KEY EXPERIMENT: Sweep OVERLAP_FEATURES from 0 to OBSERVER_DIM.
    At overlap=0: agents have completely independent states. Channel MUST carry all info.
    At overlap=OBSERVER_DIM: old tautological setup. Channel is redundant.
    """
    overlaps = list(range(0, OBSERVER_DIM + 1, 2))  # 0, 2, 4, 6, 8, 10
    results = []
    for overlap in overlaps:
        private = OBSERVER_DIM - overlap
        print(f"\n  --- overlap={overlap}, private={private} ---")
        obs = Observer(); nav = Navigator(); base = Baseline()

        # Rebuild Oracle with correct input dim per overlap
        class DynamicOracle(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(NAVIGATOR_DIM + overlap, HIDDEN), nn.ReLU(),
                    nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                    nn.Linear(HIDDEN, OBSERVER_DIM),
                )
            def forward(self, local, shared):
                if overlap == 0:
                    return self.net(local)  # no shared features to concat
                return self.net(torch.cat([local, shared], dim=-1))

        ora = DynamicOracle()

        hist = train(obs, nav, base, ora, epochs=4000, overlap=overlap)
        eval_r = evaluate_full(obs, nav, base, ora, n=200, overlap=overlap)
        eval_r["overlap"] = overlap
        results.append(eval_r)

        print(f"  MSE={eval_r['mse']:.6f}  Baseline={eval_r.get('baseline_mse', 0):.6f}  "
              f"Oracle={eval_r.get('oracle_mse', 0):.6f}  MI Gain={eval_r.get('mi_gain', 0):.1%}  "
              f"Shared corr={eval_r.get('shared_avg_corr', 0):.3f}  "
              f"Private corr={eval_r.get('private_avg_corr', 0):.3f}")

    # Plot sweep
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    overlaps_x = [r["overlap"] for r in results]
    neuralese_mse = [r["mse"] for r in results]
    baseline_mse = [r.get("baseline_mse", 0) for r in results]
    oracle_mse = [r.get("oracle_mse", 0) for r in results]
    mi_gains = [r.get("mi_gain", 0) * 100 for r in results]  # as percentage
    shared_c = [r.get("shared_avg_corr", 0) for r in results]
    private_c = [r.get("private_avg_corr", 0) for r in results]

    ax = axes[0, 0]
    ax.plot(overlaps_x, neuralese_mse, 'b-o', label='Neuralese (z channel)', markersize=8)
    ax.plot(overlaps_x, baseline_mse, 'r-s', label='Baseline (local only)', markersize=8)
    ax.plot(overlaps_x, oracle_mse, 'g-^', label='Oracle (sees shared)', markersize=8)
    ax.set_xlabel("Overlap Features (shared)")
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title("Reconstruction Error vs Overlap")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(overlaps_x, mi_gains, color=['red' if g < 10 else 'orange' if g < 50 else 'green' for g in mi_gains], alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel("Overlap Features")
    ax.set_ylabel("MI Gain (%)")
    ax.set_title("MI Gain (how much z helps over baseline)")
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    x = np.arange(len(overlaps_x))
    width = 0.35
    ax.bar(x - width/2, shared_c, width, label='Shared features (r)', color='green', alpha=0.7)
    ax.bar(x + width/2, private_c, width, label='Private features (r)', color='red', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(overlaps_x)
    ax.set_xlabel("Overlap Features"); ax.set_ylabel("Correlation (r)")
    ax.set_title("Per-Feature Correlation: Shared vs Private")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)

    ax = axes[1, 1]
    text = (f"Latent dim: {LATENT_DIM}D\n"
            f"Observer state: {OBSERVER_DIM}D\n"
            f"Navigator state: {NAVIGATOR_DIM}D\n\n"
            f"KEY FINDING:\n"
            f"At overlap=0 (independent agents),\n"
            f"MI Gain = {mi_gains[0]:.1f}%\n"
            f"Neuralese MSE = {neuralese_mse[0]:.6f}\n"
            f"Oracle MSE = {oracle_mse[0]:.6f}\n\n"
            f"At overlap={OBSERVER_DIM} (shared state),\n"
            f"MI Gain = {mi_gains[-1]:.1f}%\n"
            f"(Old confounded result)")
    ax.text(0.1, 0.5, text, fontfamily='monospace', fontsize=10, va='center', transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle(f"Neuralese Path C (FIXED) — Overlap Sweep ({LATENT_DIM}D channel)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "path_c_sweep.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"\n  Saved: {fname}")
    return results


def plot_results(hist, eval_r, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(hist["mse"], alpha=0.5, color='blue', label='Neuralese')
    if "baseline_mse" in hist:
        ax.plot(hist["baseline_mse"], alpha=0.3, color='red', label='No-z baseline')
    if "oracle_mse" in hist:
        ax.plot(hist["oracle_mse"], alpha=0.3, color='green', label='Oracle (sees shared)')
    ax.legend(); ax.set_title("Reconstruction MSE"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "mi_gain" in hist:
        ax.plot(hist["mi_gain"], alpha=0.5, color='green')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.set_title(f"MI Gain (genuine information throughput)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    corrs = eval_r["per_feature_corr"]
    bar_colors = []
    for i, c in enumerate(corrs):
        if i < OVERLAP_FEATURES:
            bar_colors.append('green')  # Shared features
        else:
            bar_colors.append('red')     # Private features
    ax.bar(range(len(corrs)), corrs, color=bar_colors, alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Strong')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.3, label='Weak')
    ax.set_title(f"Per-Feature Correlation (green=shared, red=private) avg={eval_r['avg_correlation']:.3f}")
    ax.set_xlabel("Feature"); ax.set_ylabel("r"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    text = (f"MSE: {eval_r['mse']:.6f}\n"
            f"Baseline MSE: {eval_r.get('baseline_mse', 0):.6f}\n"
            f"Oracle MSE: {eval_r.get('oracle_mse', 0):.6f}\n"
            f"MI Gain: {eval_r.get('mi_gain', 0):.1%}\n"
            f"Avg Correlation: {eval_r['avg_correlation']:.3f}\n"
            f"Shared feat corr: {eval_r.get('shared_avg_corr', 0):.3f}\n"
            f"Private feat corr: {eval_r.get('private_avg_corr', 0):.3f}\n"
            f"Neuralese/Oracle ratio: {eval_r.get('neuralese_vs_oracle', 0):.2f}x\n"
            f"\nOverlap: {OVERLAP_FEATURES}/{OBSERVER_DIM} shared features\n"
            f"Bottleneck: {OBSERVER_DIM}D → {LATENT_DIM}D → {OBSERVER_DIM}D")
    ax.text(0.1, 0.5, text, fontfamily='monospace', fontsize=10, va='center', transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle(f"Neuralese Path C (FIXED) — State Sync ({LATENT_DIM}D, overlap={OVERLAP_FEATURES})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "path_c_results.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEURALESE PATH C (FIXED) — Independent Agent State Synchronization")
    print("=" * 70)
    print(f"  OVERLAP_FEATURES: {OVERLAP_FEATURES} (shared ground truth)")
    print(f"  PRIVATE_FEATURES: {PRIVATE_FEATURES} (independent per agent)")
    print(f"  Agent A: {OBSERVER_DIM}D → {LATENT_DIM}D z")
    print(f"  Agent B: z + {NAVIGATOR_DIM}D local → {OBSERVER_DIM}D reconstruction")
    print(f"  Baseline: Agent B without z (local features only)")
    print(f"  Oracle: Agent B with local features + shared ground truth (upper bound)")
    print()

    # 1. Train with current overlap setting
    print("[1/2] Training with overlap={}...".format(OVERLAP_FEATURES))
    observer = Observer(); navigator = Navigator()
    baseline = Baseline(); oracle = Oracle(overlap=OVERLAP_FEATURES)
    hist = train(observer, navigator, baseline, oracle)

    print("\n[Evaluation]...")
    eval_r = evaluate_full(observer, navigator, baseline, oracle, n=500)
    print(f"  Neuralese MSE:       {eval_r['mse']:.6f}")
    print(f"  Baseline MSE (no z): {eval_r.get('baseline_mse', 0):.6f}")
    print(f"  Oracle MSE (ideal):  {eval_r.get('oracle_mse', 0):.6f}")
    print(f"  MI Gain:             {eval_r.get('mi_gain', 0):.1%}")
    print(f"  Avg Correlation:     {eval_r['avg_correlation']:.3f}")
    print(f"  Shared feat corr:    {eval_r.get('shared_avg_corr', 0):.3f}")
    print(f"  Private feat corr:   {eval_r.get('private_avg_corr', 0):.3f}")
    print(f"  Neuralese/Oracle:    {eval_r.get('neuralese_vs_oracle', 0):.2f}x worse than oracle")

    plot_results(hist, eval_r, out_dir)

    # 2. Sweep overlap parameter
    print("\n\n[2/2] OVERLAP SWEEP — The definitive test")
    print("=" * 70)
    print("Sweeping OVERLAP_FEATURES from 0 (fully independent agents) to "
          f"{OBSERVER_DIM} (fully shared = old confounded setup)")
    sweep_results = sweep_overlap(out_dir)

    print("\n\n" + "=" * 70)
    print("OVERLAP SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Overlap':>8} {'MSE':>10} {'Baseline':>10} {'Oracle':>10} {'MI Gain':>10} {'N/O ratio':>10}")
    print("-" * 70)
    for r in sweep_results:
        print(f"{r['overlap']:>8} {r['mse']:>10.6f} {r.get('baseline_mse',0):>10.6f} "
              f"{r.get('oracle_mse',0):>10.6f} {r.get('mi_gain',0):>9.1%} "
              f"{r.get('neuralese_vs_oracle',0):>9.2f}x")

    print(f"\nDone! Outputs in {out_dir}/")
