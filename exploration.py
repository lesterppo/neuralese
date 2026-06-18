"""
Neuralese Referential Game — Systematic Exploration
====================================================
Explores the boundaries of the emergent communication protocol:
  1. BOTTLENECK SWEEP: 2D → 32D. Find minimum viable dimension.
  2. CANDIDATE SCALING: 4 → 32 candidates. How many can the Receiver distinguish?
  3. INFORMATION THROUGHPUT: bits per correct identification.
  4. PROTOCOL ANALYSIS: What information does the latent vector encode?
"""

import torch, torch.nn as nn, torch.optim as optim, torch.distributions as D
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from referential_game import (
    gen_function, func_to_embedding, FunctionBank,
    Sender as _Sender, Receiver as _Receiver,
    EMBED_DIM, HIDDEN
)

# --- DYNAMIC CONFIG ---
EPOCHS_PER_RUN = 4000  # Reduced for speed in sweep
BATCH = 32
LR = 1e-3
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
TRAIN_FUNCS = 600
TEST_FUNCS = 200

# --- DYNAMIC MODELS ---

def make_sender(latent_dim):
    """Dynamically create Sender with given bottleneck."""
    class DynSender(nn.Module):
        def __init__(self):
            super().__init__()
            self.char_embed = nn.Embedding(128, 32)
            self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
            self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
            self.conv3 = nn.Conv1d(64, 64, 7, padding=3)
            self.fc = nn.Sequential(nn.Linear(192, HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
            self.z_head = nn.Sequential(nn.Linear(HIDDEN, latent_dim),
                                        nn.LayerNorm(latent_dim))
            self.v_head = nn.Linear(HIDDEN, 1)

        def forward(self, func_texts):
            B = len(func_texts)
            L = min(max(len(t) for t in func_texts), 300)
            ids = torch.zeros(B, L, dtype=torch.long)
            for b, t in enumerate(func_texts):
                for i, ch in enumerate(t[:L]):
                    ids[b, i] = min(ord(ch) % 128, 127)
            emb = self.char_embed(ids).permute(0, 2, 1)
            c1 = torch.relu(self.conv1(emb))
            c2 = torch.relu(self.conv2(c1))
            c3 = torch.relu(self.conv3(c2))
            feat = torch.cat([c1.mean(-1), c2.mean(-1), c3.mean(-1)], dim=-1)
            h = self.fc(feat)
            return self.z_head(h), self.v_head(h).squeeze(-1)
    return DynSender()


def make_receiver(latent_dim, num_candidates):
    """Dynamically create Receiver with given params."""
    class DynReceiver(nn.Module):
        def __init__(self):
            super().__init__()
            self.cand_net = nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN), nn.ReLU())
            self.scorer = nn.Sequential(
                nn.Linear(HIDDEN + latent_dim, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, 1))

        def forward(self, candidates, z):
            B, N, _ = candidates.shape
            h = self.cand_net(candidates.view(B*N, EMBED_DIM))
            z_exp = z.unsqueeze(1).expand(B, N, latent_dim).reshape(B*N, latent_dim)
            scores = self.scorer(torch.cat([h, z_exp], -1))
            return scores.view(B, N)
    return DynReceiver()


# --- TRAINING ---

def train_one(sender, receiver, bank, latent_dim, num_candidates, epochs=EPOCHS_PER_RUN):
    params = list(sender.parameters()) + list(receiver.parameters())
    opt = optim.Adam(params, lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0.0

    for ep in range(epochs):
        candidates, target_pos, target_texts = bank.sample_game(BATCH)

        z, values = sender(target_texts)
        logits = receiver(candidates, z)

        dist = D.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        rewards = (actions == target_pos).float() * 2.0 - 1.0
        entropy = dist.entropy().mean()

        advantage = rewards - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = nn.MSELoss()(values, rewards)
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        acc = (actions == target_pos).float().mean().item()
        if acc > best_acc: best_acc = acc

    return best_acc


# --- EVALUATION ---

def evaluate_one(sender, receiver, bank, latent_dim, num_candidates, n_games=300):
    correct = 0; null_correct = 0
    for _ in range(n_games):
        candidates, target_pos, target_texts = bank.sample_game(1)
        with torch.no_grad():
            z, _ = sender(target_texts)
            pred = receiver(candidates, z).argmax(-1).item()
            z_null = torch.randn(1, latent_dim) * 0.5
            pred_null = receiver(candidates, z_null).argmax(-1).item()
        if pred == target_pos[0].item(): correct += 1
        if pred_null == target_pos[0].item(): null_correct += 1
    return {
        "accuracy": correct / n_games,
        "null_accuracy": null_correct / n_games,
        "chance": 1.0 / num_candidates,
        "over_chance": correct / n_games - 1.0 / num_candidates,
    }


# --- EXPERIMENT 1: BOTTLENECK SWEEP ---

def experiment_bottleneck_sweep(out_dir):
    """Sweep LATENT_DIM from 2 to 32 with NUM_CANDIDATES=4."""
    print("=" * 70)
    print("EXPERIMENT 1: BOTTLENECK SWEEP")
    print("=" * 70)
    print("  Question: What is the minimum bottleneck dimension for communication?")
    print("  Candidates fixed at 4 (chance=25%)")
    print()

    dims = [2, 4, 6, 8, 12, 16, 24, 32]
    results = []

    for ld in dims:
        print(f"  {ld}D bottleneck...", end=" ", flush=True)
        train_bank = FunctionBank(TRAIN_FUNCS)
        test_bank = FunctionBank(TEST_FUNCS)

        sender = make_sender(ld)
        receiver = make_receiver(ld, 4)

        best_train = train_one(sender, receiver, train_bank, ld, 4)
        eval_r = evaluate_one(sender, receiver, test_bank, ld, 4)

        # Information throughput: log2(candidates) bits per correct identification
        bits_per_game = np.log2(4)  # 2 bits per 4-choice game
        throughput = eval_r["accuracy"] * bits_per_game
        bits_per_dim = throughput / ld if ld > 0 else 0

        r = {
            "dim": ld,
            "best_train": best_train,
            "test_acc": eval_r["accuracy"],
            "null_acc": eval_r["null_accuracy"],
            "over_chance": eval_r["over_chance"],
            "throughput_bits": throughput,
            "bits_per_dim": bits_per_dim,
        }
        results.append(r)
        print(f"best={best_train:.1%} test={eval_r['accuracy']:.1%} "
              f"null={eval_r['null_accuracy']:.1%} bits/dim={bits_per_dim:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dx = [r["dim"] for r in results]

    ax = axes[0, 0]
    ax.plot(dx, [r["test_acc"] for r in results], 'b-o', label='Test accuracy', markersize=8)
    ax.plot(dx, [r["null_acc"] for r in results], 'r--s', label='Null channel', markersize=8)
    ax.axhline(y=0.25, color='gray', ls=':', alpha=0.5, label='Chance (25%)')
    ax.axhline(y=0.50, color='green', ls='--', alpha=0.3, label='50% threshold')
    ax.set_xlabel("Bottleneck dimension"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Bottleneck Size"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(dx, [r["over_chance"] for r in results], 'purple', marker='o', markersize=8)
    ax.fill_between(dx, 0, [r["over_chance"] for r in results], alpha=0.2, color='purple')
    ax.axhline(y=0, color='red', ls='-', alpha=0.3)
    ax.axhline(y=0.05, color='orange', ls='--', alpha=0.3, label='5% significance')
    ax.set_xlabel("Bottleneck dimension"); ax.set_ylabel("Over chance")
    ax.set_title("Signal Above Chance"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(dx, [r["throughput_bits"] for r in results], 'green', marker='s', markersize=8, label='Throughput')
    ax2 = ax.twinx()
    ax2.plot(dx, [r["bits_per_dim"] for r in results], 'orange', marker='^', markersize=8, label='Bits per dim')
    ax.set_xlabel("Bottleneck dimension"); ax.set_ylabel("Bits per game")
    ax2.set_ylabel("Bits per dimension")
    ax.set_title("Information Throughput")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    summary = "BOTTLENECK SWEEP\n" + "="*20 + "\n\n"
    for r in results:
        marker = "✓" if r["over_chance"] > 0.05 else " "
        summary += f"{r['dim']:2d}D: {r['test_acc']:.1%} {marker}\n"
    summary += f"\nMinimum viable: "
    viable = [r for r in results if r["over_chance"] > 0.05]
    summary += f"{viable[0]['dim']}D" if viable else "none found"
    ax.text(0.1, 0.5, summary, fontfamily='monospace', fontsize=9, va='center',
            transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle("Neuralese Bottleneck Sweep", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "exploration_bottleneck.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {out_dir}/exploration_bottleneck.png")

    return results


# --- EXPERIMENT 2: CANDIDATE SCALING ---

def experiment_candidate_scaling(out_dir, latent_dim=16):
    """Scale NUM_CANDIDATES from 4 to 32 with fixed 16D bottleneck."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CANDIDATE SCALING")
    print("=" * 70)
    print(f"  Question: How many candidates can 16D distinguish?")
    print(f"  Bottleneck fixed at {latent_dim}D")
    print()

    n_cands_list = [4, 8, 16, 32]
    results = []

    for nc in n_cands_list:
        print(f"  {nc} candidates (chance={1/nc:.1%})...", end=" ", flush=True)
        # Create banks with larger candidate pools
        train_bank = FunctionBank(TRAIN_FUNCS)
        test_bank = FunctionBank(TEST_FUNCS)

        sender = make_sender(latent_dim)
        receiver = make_receiver(latent_dim, nc)

        # Temporarily override NUM_CANDIDATES for sampling
        import referential_game as rg
        old_nc = rg.NUM_CANDIDATES
        rg.NUM_CANDIDATES = nc
        train_bank_nc = FunctionBank(TRAIN_FUNCS)  # rebuild with new NC

        best_train = train_one(sender, receiver, train_bank_nc, latent_dim, nc)
        eval_r = evaluate_one(sender, receiver, test_bank, latent_dim, nc)

        bits_per_game = np.log2(nc)
        throughput = eval_r["accuracy"] * bits_per_game

        r = {
            "candidates": nc,
            "chance": 1.0 / nc,
            "best_train": best_train,
            "test_acc": eval_r["accuracy"],
            "null_acc": eval_r["null_accuracy"],
            "over_chance": eval_r["over_chance"],
            "throughput_bits": throughput,
            "bits_needed": np.log2(nc),  # theoretical minimum bits
        }

        rg.NUM_CANDIDATES = old_nc  # restore

        results.append(r)
        print(f"best={best_train:.1%} test={eval_r['accuracy']:.1%} "
              f"null={eval_r['null_accuracy']:.1%} thr={throughput:.2f} bits")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cx = [r["candidates"] for r in results]

    ax = axes[0, 0]
    ax.plot(cx, [r["test_acc"] for r in results], 'b-o', label='Test accuracy', markersize=8)
    ax.plot(cx, [r["chance"] for r in results], 'r--s', label='Chance', markersize=8)
    ax.plot(cx, [r["null_acc"] for r in results], 'gray', marker='x', label='Null', markersize=8)
    ax.set_xlabel("Number of candidates"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Set Size"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    ax = axes[0, 1]
    ax.plot(cx, [r["throughput_bits"] for r in results], 'green', marker='s', markersize=8)
    ax.plot(cx, [r["bits_needed"] for r in results], 'green', ls='--', alpha=0.5, label='Max possible')
    ax.set_xlabel("Number of candidates"); ax.set_ylabel("Bits per game")
    ax.set_title("Information Throughput"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    ax = axes[1, 0]
    # Efficiency: actual bits / theoretical max
    efficiencies = [r["throughput_bits"] / max(r["bits_needed"], 1e-8) for r in results]
    ax.bar(range(len(cx)), efficiencies, color=['green' if e > 0.5 else 'orange' if e > 0.2 else 'red' for e in efficiencies], alpha=0.7)
    ax.set_xticks(range(len(cx))); ax.set_xticklabels(cx)
    ax.set_xlabel("Candidates"); ax.set_ylabel("Efficiency")
    ax.set_title(f"Channel Efficiency ({latent_dim}D bottleneck)"); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='green', ls='--', alpha=0.3)

    ax = axes[1, 1]
    summary = f"CANDIDATE SCALING\n{'='*20}\n{latent_dim}D bottleneck\n\n"
    for r in results:
        summary += f"{r['candidates']:2d} cand: {r['test_acc']:.1%}\n"
    best_nc = max([r for r in results if r["over_chance"] > 0.05], key=lambda r: r["candidates"], default=None)
    summary += f"\nMax distinguishable: {best_nc['candidates'] if best_nc else 'none'} candidates"
    ax.text(0.1, 0.5, summary, fontfamily='monospace', fontsize=9, va='center',
            transform=ax.transAxes)
    ax.set_title("Summary"); ax.axis('off')

    plt.suptitle(f"Neuralese Candidate Scaling ({latent_dim}D)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "exploration_candidates.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {out_dir}/exploration_candidates.png")

    return results


# --- EXPERIMENT 3: PROTOCOL ANALYSIS ---

def experiment_protocol_analysis(sender, receiver, bank, latent_dim, out_dir):
    """Analyze what the latent vector encodes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PROTOCOL ANALYSIS")
    print("=" * 70)

    # Collect latent vectors for different function types
    func_categories = {
        "dict_merge": [],
        "list_filter": [],
        "string_op": [],
        "file_io": [],
        "math_compute": [],
    }
    z_vectors = {k: [] for k in func_categories}

    for _ in range(200):
        f = gen_function()
        for cat, keywords in [
            ("dict_merge", ["dict", "merge", "update"]),
            ("list_filter", ["filter", "list", "items"]),
            ("string_op", ["split", "join", "case", "text"]),
            ("file_io", ["open", "read", "write", "json", "csv", "path"]),
            ("math_compute", ["sum", "mean", "sort", "compute", "+", "*"]),
        ]:
            if any(kw in f for kw in keywords):
                with torch.no_grad():
                    z, _ = sender([f])
                z_vectors[cat].append(z.squeeze(0).numpy())
                break

    # Compute per-category mean z vectors
    cat_means = {}
    for cat, vecs in z_vectors.items():
        if vecs:
            cat_means[cat] = np.mean(vecs, axis=0)

    if len(cat_means) < 2:
        print("  Not enough category diversity for analysis")
        return None

    # Plot category centroids
    cats = list(cat_means.keys())
    centroids = np.array([cat_means[c] for c in cats])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of category centroids
    ax = axes[0]
    im = ax.imshow(centroids, cmap='RdBu_r', aspect='auto')
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xlabel("Latent dimension"); ax.set_title("Category Centroids")
    plt.colorbar(im, ax=ax)

    # Pairwise distances between categories
    from scipy.spatial.distance import pdist, squareform
    if len(cats) >= 2:
        dists = squareform(pdist(centroids))
        ax = axes[1]
        im = ax.imshow(dists, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=45)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
        ax.set_title("Category Distances")
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, f"{dists[i,j]:.2f}", ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Neuralese Protocol Analysis — Category Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "exploration_protocol.png", dpi=150)
    plt.close()
    print(f"  Categories: {list(cats)}")
    if len(cats) >= 2:
        print(f"  Pairwise distances:\n{dists}")
    print(f"  Saved: {out_dir}/exploration_protocol.png")
    return {"categories": cats, "centroids": centroids, "distances": dists if len(cats) >= 2 else None}


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEURALESE SYSTEMATIC EXPLORATION")
    print("=" * 70)
    print(f"  Epochs per run: {EPOCHS_PER_RUN}")
    print(f"  Train funcs: {TRAIN_FUNCS}, Test funcs: {TEST_FUNCS}")

    # Experiment 1: Bottleneck sweep
    bottleneck_results = experiment_bottleneck_sweep(out_dir)

    # Experiment 2: Candidate scaling (use 16D based on sweep results)
    print()
    candidate_results = experiment_candidate_scaling(out_dir, latent_dim=16)

    # Experiment 3: Protocol analysis (retrain at 16D, 4 candidates)
    print()
    train_bank = FunctionBank(TRAIN_FUNCS)
    sender = make_sender(16)
    receiver = make_receiver(16, 4)
    train_one(sender, receiver, train_bank, 16, 4, epochs=EPOCHS_PER_RUN)
    protocol_results = experiment_protocol_analysis(sender, receiver, train_bank, 16, out_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("EXPLORATION SUMMARY")
    print("=" * 70)

    print("\nBOTTLENECK SWEEP:")
    print(f"{'Dim':>5} {'Test Acc':>10} {'Null':>10} {'Over Chance':>13}")
    for r in bottleneck_results:
        print(f"{r['dim']:>5} {r['test_acc']:>9.1%} {r['null_acc']:>9.1%} {r['over_chance']:>12.1%}")

    print("\nCANDIDATE SCALING (16D):")
    print(f"{'Cands':>7} {'Test Acc':>10} {'Chance':>10} {'Throughput':>12}")
    for r in candidate_results:
        print(f"{r['candidates']:>7} {r['test_acc']:>9.1%} {r['chance']:>9.1%} {r['throughput_bits']:>11.2f} bits")

    print("\nDone! Outputs in output/")
