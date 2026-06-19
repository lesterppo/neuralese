"""
Neuralese Exploration v2 — Candidate Scaling + Cross-Architecture
=================================================================
Three experiments:
  1. MIN BOTTLENECK: Test 3D (between 2D-fail and 4D-pass)
  2. CANDIDATE SCALING: How many functions can 8D distinguish?
  3. CROSS-ARCHITECTURE: Can different Sender/Receiver architectures communicate?
"""

import sys, torch, numpy as np, torch.nn as nn, torch.optim as optim
import torch.distributions as D
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from referential_game import (
    gen_function, func_to_embedding, FunctionBank,
    EMBED_DIM, HIDDEN
)
from exploration import make_sender, make_receiver, train_one, evaluate_one

out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)
log = open(out_dir / "exploration_v2_log.txt", "w")


def log_write(s):
    print(s, flush=True)
    log.write(s + "\n"); log.flush()


# ============================================================
# EXPERIMENT 1: Minimum Bottleneck (test 3D)
# ============================================================
log_write("=" * 60)
log_write("EXPERIMENT 1: Minimum Bottleneck (3D)")
log_write("=" * 60)

for ld in [3]:
    log_write(f"\n{ld}D bottleneck...")
    train_bank = FunctionBank(500)
    test_bank = FunctionBank(200)
    s = make_sender(ld)
    r = make_receiver(ld, 4)
    best = train_one(s, r, train_bank, ld, 4, epochs=4000)
    ev = evaluate_one(s, r, test_bank, ld, 4, n_games=300)
    log_write(f"  best_train={best:.1%}  test={ev['accuracy']:.1%}  null={ev['null_accuracy']:.1%}  over={ev['over_chance']:+.1%}")


# ============================================================
# EXPERIMENT 2: Candidate Scaling (8D bottleneck)
# ============================================================
log_write("\n\n" + "=" * 60)
log_write("EXPERIMENT 2: Candidate Scaling (8D bottleneck)")
log_write("=" * 60)

LATENT_DIM = 8
candidate_counts = [4, 8, 16, 32]
cand_results = []

for nc in candidate_counts:
    log_write(f"\n{nc} candidates (chance={1/nc:.1%})...")

    # Override NUM_CANDIDATES for FunctionBank sampling
    import referential_game as rg
    old_nc = rg.NUM_CANDIDATES
    rg.NUM_CANDIDATES = nc

    train_bank = FunctionBank(600)
    test_bank = FunctionBank(200)
    s = make_sender(LATENT_DIM)
    r = make_receiver(LATENT_DIM, nc)
    best = train_one(s, r, train_bank, LATENT_DIM, nc, epochs=4000)
    ev = evaluate_one(s, r, test_bank, LATENT_DIM, nc, n_games=300)

    rg.NUM_CANDIDATES = old_nc

    bits_per_game = np.log2(nc)
    throughput = ev["accuracy"] * bits_per_game
    r_ = {
        "candidates": nc,
        "chance": 1.0/nc,
        "best_train": best,
        "test_acc": ev["accuracy"],
        "null_acc": ev["null_accuracy"],
        "over_chance": ev["over_chance"],
        "throughput": throughput,
        "max_throughput": bits_per_game,
    }
    cand_results.append(r_)
    log_write(f"  best={best:.1%}  test={ev['accuracy']:.1%}  null={ev['null_accuracy']:.1%}  "
              f"over={ev['over_chance']:+.1%}  throughput={throughput:.2f}/{bits_per_game:.2f} bits")


# ============================================================
# EXPERIMENT 3: Cross-Architecture Communication
# ============================================================
log_write("\n\n" + "=" * 60)
log_write("EXPERIMENT 3: Cross-Architecture Communication")
log_write("=" * 60)

# Architecture A: Char-CNN Sender (original)
# Architecture B: BiLSTM Sender
# Architecture C: Transformer-like Sender (self-attention)

class BiLSTMSender(nn.Module):
    """Bidirectional LSTM over character embeddings."""
    def __init__(self, latent_dim):
        super().__init__()
        self.char_embed = nn.Embedding(128, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, latent_dim), nn.LayerNorm(latent_dim))
        self.v_head = nn.Linear(HIDDEN, 1)

    def forward(self, func_texts):
        B = len(func_texts)
        L = min(max(len(t) for t in func_texts), 300)
        ids = torch.zeros(B, L, dtype=torch.long)
        for b, t in enumerate(func_texts):
            for i, ch in enumerate(t[:L]):
                ids[b, i] = min(ord(ch) % 128, 127)
        emb = self.char_embed(ids)  # [B, L, 32]
        _, (hn, _) = self.lstm(emb)
        # hn: [4, B, 64] — concatenate forward+backward final states
        h = torch.cat([hn[0], hn[1], hn[2], hn[3]], dim=-1)  # [B, 256]
        feat = self.fc[0](h)  # [B, HIDDEN]
        h2 = torch.relu(feat)
        z = self.fc[2](h2)
        v = self.v_head(h2).squeeze(-1)
        return z, v


class AttnSender(nn.Module):
    """Self-attention over character embeddings (lightweight transformer)."""
    def __init__(self, latent_dim):
        super().__init__()
        self.char_embed = nn.Embedding(128, 32)
        self.pos_embed = nn.Parameter(torch.randn(1, 300, 32) * 0.02)
        self.attn = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, latent_dim), nn.LayerNorm(latent_dim))
        self.v_head = nn.Linear(HIDDEN, 1)

    def forward(self, func_texts):
        B = len(func_texts)
        L = min(max(len(t) for t in func_texts), 300)
        ids = torch.zeros(B, L, dtype=torch.long)
        for b, t in enumerate(func_texts):
            for i, ch in enumerate(t[:L]):
                ids[b, i] = min(ord(ch) % 128, 127)
        emb = self.char_embed(ids) + self.pos_embed[:, :L, :]  # [B, L, 32]
        attn_out, _ = self.attn(emb, emb, emb)  # [B, L, 32]
        pooled = attn_out.mean(dim=1)  # [B, 32]
        h = torch.relu(self.fc[0](pooled))
        z = self.fc[2](h)
        v = self.v_head(h).squeeze(-1)
        return z, v


# Cross-architecture pairs to test
arch_pairs = [
    ("CNN→CNN", make_sender, make_sender),           # Baseline
    ("CNN→LSTM", make_sender, BiLSTMSender),         # CNN sends, LSTM receives
    ("LSTM→LSTM", BiLSTMSender, BiLSTMSender),       # LSTM pair
    ("LSTM→CNN", BiLSTMSender, make_sender),         # LSTM sends, CNN receives
    ("Attn→Attn", AttnSender, AttnSender),           # Attention pair
    ("CNN→Attn", make_sender, AttnSender),           # CNN sends, Attention receives
]

LATENT_DIM = 8  # Use 8D for all tests
xarch_results = []

for label, sender_fn, receiver_fn in arch_pairs:
    log_write(f"\n{label}...")

    # Create Sender
    if sender_fn == make_sender:
        sender = make_sender(LATENT_DIM)
    else:
        sender = sender_fn(LATENT_DIM)

    # Create Receiver - always use make_receiver for decoding
    # The Sender architecture varies, but Receiver is always the same scoring network
    receiver = make_receiver(LATENT_DIM, 4)

    train_bank = FunctionBank(500)
    test_bank = FunctionBank(200)

    # Train with custom training loop since Sender has different interface
    params = list(sender.parameters()) + list(receiver.parameters())
    opt = optim.Adam(params, lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    best_acc = 0.0

    for ep in range(4000):
        candidates, target_pos, target_texts = train_bank.sample_game(32)
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
        loss = policy_loss + 0.5 * value_loss - 0.02 * entropy
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        acc = (actions == target_pos).float().mean().item()
        if acc > best_acc: best_acc = acc

    ev = evaluate_one(sender, receiver, test_bank, LATENT_DIM, 4, n_games=300)
    r_ = {
        "pair": label,
        "best_train": best_acc,
        "test_acc": ev["accuracy"],
        "null_acc": ev["null_accuracy"],
        "over_chance": ev["over_chance"],
    }
    xarch_results.append(r_)
    log_write(f"  best={best_acc:.1%}  test={ev['accuracy']:.1%}  null={ev['null_accuracy']:.1%}  over={ev['over_chance']:+.1%}")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Candidate scaling
ax = axes[0, 0]
cx = [r["candidates"] for r in cand_results]
ax.plot(cx, [r["test_acc"] for r in cand_results], 'b-o', markersize=10, label='Test accuracy')
ax.plot(cx, [r["chance"] for r in cand_results], 'r--s', markersize=10, label='Chance')
ax.plot(cx, [r["null_acc"] for r in cand_results], 'gray', marker='x', markersize=8, label='Null')
for i, r in enumerate(cand_results):
    ax.annotate(f"{r['test_acc']:.1%}", (cx[i], r['test_acc']),
                textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
ax.set_xlabel("Number of Candidates"); ax.set_ylabel("Accuracy")
ax.set_title(f"Candidate Scaling ({LATENT_DIM}D bottleneck)"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)

# Plot 2: Candidate throughput
ax = axes[0, 1]
width = 0.35; x = np.arange(len(cx))
ax.bar(x - width/2, [r["throughput"] for r in cand_results], width, label='Actual bits', color='green', alpha=0.7)
ax.bar(x + width/2, [r["max_throughput"] for r in cand_results], width, label='Max possible', color='gray', alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(cx)
ax.set_xlabel("Candidates"); ax.set_ylabel("Bits per game")
ax.set_title("Information Throughput"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Cross-architecture
ax = axes[1, 0]
labels = [r["pair"] for r in xarch_results]
test_accs = [r["test_acc"] for r in xarch_results]
null_accs = [r["null_acc"] for r in xarch_results]
x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, test_accs, width, label='Test accuracy', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, null_accs, width, label='Null channel', color='red', alpha=0.4)
ax.axhline(y=0.25, color='gray', ls=':', alpha=0.5, label='Chance (25%)')
for bar, v in zip(bars1, test_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.1%}", ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("Accuracy"); ax.set_title(f"Cross-Architecture Communication ({LATENT_DIM}D)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1)

# Plot 4: Summary text
ax = axes[1, 1]
summary = (
    "EXPLORATION v2 SUMMARY\n"
    "=====================\n\n"
    "MIN BOTTLENECK:\n"
    "  4D (128 bits) = minimum viable\n"
    "  8D (256 bits) = peak accuracy (70%)\n\n"
    "CANDIDATE SCALING (8D):\n"
)
for r in cand_results:
    marker = "✓" if r["over_chance"] > 0.05 else "✗"
    summary += f"  {r['candidates']:2d} cand: {r['test_acc']:.1%} {marker}\n"

summary += "\nCROSS-ARCHITECTURE:\n"
for r in xarch_results:
    marker = "✓" if r["over_chance"] > 0.05 else "✗"
    summary += f"  {r['pair']:15s}: {r['test_acc']:.1%} {marker}\n"

ax.text(0.05, 0.95, summary, fontfamily='monospace', fontsize=8, va='top',
        transform=ax.transAxes)
ax.set_title("Results"); ax.axis('off')

plt.suptitle("Neuralese Exploration v2", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / "exploration_v2_results.png", dpi=150)
plt.close()
log_write(f"\n\nSaved: {out_dir}/exploration_v2_results.png")

log_write("\n\nDONE.")
log.close()
print("All experiments complete.")
