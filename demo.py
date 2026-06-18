"""
Neuralese Demo — Continuous Latent Communication vs Text Baseline
Collaborative development: Hermes (DeepSeek-v4) + Gemini Pro
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
HIDDEN_DIM = 128
BOTTLENECK_DIM = 8       # Neuralese: 8 * 32 = 256 bits
VOCAB_SIZE = 64
TOKENS_USED = 4           # Text: 4 tokens * 6 bits = 24 bits  (unfair to text!)
# Fair comparison: give text MORE tokens so bit budgets are similar
TOKENS_FAIR = 42          # ~256 bits = 42 * log2(64) → roughly matching 256 bits
EPOCHS = 5000
BATCH_SIZE = 128
LR = 1e-3
L2_LAMBDA = 0.01          # Bottleneck pressure for Neuralese

# --- NEURALESE MODELS ---
class NeuraleseAgentA(nn.Module):
    """Encodes 2D coordinate → bottleneck vector (the 'Neuralese message')"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
            nn.LayerNorm(BOTTLENECK_DIM),  # Stable distribution
        )
    def forward(self, x):
        return self.net(x)

class NeuraleseAgentB(nn.Module):
    """Decodes bottleneck vector → reconstructed coordinate"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )
    def forward(self, z):
        return self.net(z)

# --- TEXT BASELINE MODELS ---
class TextAgentA(nn.Module):
    """Encodes 2D coordinate → discrete token sequence via Gumbel-Softmax"""
    def __init__(self, tokens_used=TOKENS_USED):
        super().__init__()
        self.tokens_used = tokens_used
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, tokens_used * VOCAB_SIZE),
        )

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.net(x).view(-1, self.tokens_used, VOCAB_SIZE)
        if hard:
            # Eval mode: straight-through Gumbel-Softmax with argmax forward
            tokens = torch.argmax(logits, dim=-1)
            return tokens, logits
        # Train mode: Gumbel-Softmax for gradient flow
        soft = nn.functional.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        tokens = torch.argmax(logits, dim=-1)
        return tokens, logits, soft

class TextAgentB(nn.Module):
    """Decodes token sequence → reconstructed coordinate"""
    def __init__(self, tokens_used=TOKENS_USED):
        super().__init__()
        self.tokens_used = tokens_used
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.net = nn.Sequential(
            nn.Linear(tokens_used * HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, tokens):
        emb = self.embedding(tokens).view(-1, self.tokens_used * HIDDEN_DIM)
        return self.net(emb)

# --- TRAINING ---
def train_neuralese(agent_a, agent_b, epochs=EPOCHS):
    opt = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []
    for step in range(epochs):
        x = torch.rand(BATCH_SIZE, 2)
        z = agent_a(x)
        pred = agent_b(z)
        task_loss = nn.MSELoss()(pred, x)
        l2_loss = L2_LAMBDA * torch.norm(z, p=2)
        loss = task_loss + l2_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history.append((task_loss.item(), l2_loss.item()))
        if step % 1000 == 0:
            print(f"  Neuralese step {step:5d}: task_loss={task_loss.item():.6f} l2={l2_loss.item():.6f}")
    return history

def train_text(agent_a, agent_b, epochs=EPOCHS):
    opt = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []
    mse = nn.MSELoss()
    temp_start, temp_end = 5.0, 0.5  # Anneal temperature
    for step in range(epochs):
        x = torch.rand(BATCH_SIZE, 2)
        temp = temp_start * (temp_end / temp_start) ** (step / epochs)
        tokens, logits, soft = agent_a(x, temperature=temp)
        # Use Gumbel-Softmax for gradient flow (continuous relaxation)
        emb = torch.matmul(soft, agent_b.embedding.weight)
        pred = agent_b.net(emb.view(-1, agent_a.tokens_used * HIDDEN_DIM))
        loss = mse(pred, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history.append(loss.item())
        if step % 1000 == 0:
            with torch.no_grad():
                hard_emb = agent_b.embedding(tokens).view(-1, agent_a.tokens_used * HIDDEN_DIM)
                hard_pred = agent_b.net(hard_emb)
                hard_mse = mse(hard_pred, x).item()
            print(f"  Text step    {step:5d}: soft_loss={loss.item():.6f} hard_mse={hard_mse:.6f}")
    return history

# --- EVALUATION ---
def evaluate(agent_a, agent_b, is_text=False):
    test_x = torch.rand(500, 2)
    with torch.no_grad():
        if not is_text:
            z = agent_a(test_x)
            pred = agent_b(z)
        else:
            tokens, _ = agent_a(test_x, hard=True)
            pred = agent_b(tokens)
        mse = nn.MSELoss()(pred, test_x).item()
    return mse, test_x

def compute_bandwidth(is_text=False, tokens_used=TOKENS_USED):
    if is_text:
        return tokens_used * np.log2(VOCAB_SIZE)
    else:
        return BOTTLENECK_DIM * 32

# --- VISUALIZATION ---
def plot_results(neur_history, text_histories, neur_mse, text_mses, test_x, neur_a):
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1. Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Neuralese loss
    task_losses = [h[0] for h in neur_history]
    l2_losses = [h[1] for h in neur_history]
    ax = axes[0]
    ax.plot(task_losses, alpha=0.5, label="Task MSE", color="blue")
    ax.plot(l2_losses, alpha=0.5, label="L2 bottleneck", color="red")
    ax.set_title("Neuralese Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_yscale("log")

    # Text baseline losses (all variants)
    ax = axes[1]
    for label, hist in text_histories.items():
        ax.plot(hist, alpha=0.6, label=label)
    ax.set_title("Text Baseline Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_yscale("log")

    # Bandwidth vs MSE comparison
    ax = axes[2]
    neur_bits = compute_bandwidth(is_text=False)
    neur_label = f"Neuralese\n({BOTTLENECK_DIM}×32bit={neur_bits} bits)"
    ax.scatter([neur_bits], [neur_mse], s=200, c="blue", zorder=5, label=neur_label)

    colors = ["red", "orange", "green"]
    for i, (label, mse) in enumerate(text_mses.items()):
        # Parse "text_Ntokens" → N
        tok = int(label.replace("text_", "").replace("tokens", ""))
        bits = compute_bandwidth(is_text=True, tokens_used=tok)
        ax.scatter([bits], [mse], s=100, c=colors[i % len(colors)], zorder=4,
                   label=f"{label}\n({tok}×{np.log2(VOCAB_SIZE):.0f}bit={bits:.0f} bits)")

    ax.set_xlabel("Bits per Message")
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title("Bandwidth Efficiency\n(further left + lower = better)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, max(neur_bits + 50, 300))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "neuralese_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'neuralese_results.png'}")

    # 2. t-SNE visualization
    z_space = neur_a(test_x).detach().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z_space)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color by x coordinate
    sc1 = axes[0].scatter(tsne[:, 0], tsne[:, 1], c=test_x[:, 0].numpy(), cmap="coolwarm", s=15)
    axes[0].set_title("Neuralese Latent Space\n(colored by X coordinate)")
    plt.colorbar(sc1, ax=axes[0], label="Input X")

    # Color by y coordinate
    sc2 = axes[1].scatter(tsne[:, 0], tsne[:, 1], c=test_x[:, 1].numpy(), cmap="coolwarm", s=15)
    axes[1].set_title("Neuralese Latent Space\n(colored by Y coordinate)")
    plt.colorbar(sc2, ax=axes[1], label="Input Y")

    plt.tight_layout()
    fig.savefig(out_dir / "neuralese_tsne.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'neuralese_tsne.png'}")

    # 3. Interpolation test
    # Train on [0,1] range, test on held-out precise coordinates
    interp_x = torch.tensor([
        [0.1234, 0.5678], [0.3333, 0.7777], [0.5555, 0.1234],
        [0.7777, 0.8888], [0.9999, 0.0001], [0.4321, 0.8765],
        [0.1111, 0.2222], [0.6666, 0.4444], [0.8888, 0.3333],
        [0.2468, 0.1357],
    ], dtype=torch.float32)
    with torch.no_grad():
        z_interp = neur_a(interp_x)
        pred_interp = neur_b(z_interp)
        mse_interp = nn.MSELoss()(pred_interp, interp_x).item()

    return mse_interp

# --- MAIN ---
if __name__ == "__main__":
    print("=" * 60)
    print("NEURALESE DEMO — Continuous vs Discrete Communication")
    print("=" * 60)

    # 1. Train Neuralese
    print("\n[1/3] Training Neuralese (continuous latent channel)...")
    neur_a, neur_b = NeuraleseAgentA(), NeuraleseAgentB()
    neur_history = train_neuralese(neur_a, neur_b)

    # 2. Train Text Baselines (multiple token budgets)
    print("\n[2/3] Training Text Baselines (discrete token channel)...")
    text_configs = {
        "text_4tokens": 4,     # 4 * 6 = 24 bits (unfair — too low)
        "text_21tokens": 21,   # 21 * 6 = 126 bits (half Neuralese)
        "text_42tokens": 42,   # 42 * 6 = 252 bits (matching Neuralese)
    }

    text_histories = {}
    text_mses = {}
    text_models = {}

    for label, ntoks in text_configs.items():
        print(f"\n  Training {label} ({ntoks} tokens)...")
        ta, tb = TextAgentA(tokens_used=ntoks), TextAgentB(tokens_used=ntoks)
        hist = train_text(ta, tb)
        text_histories[label] = hist
        text_models[label] = (ta, tb)

    # 3. Evaluate
    print("\n[3/3] Evaluating...")
    neur_mse, test_x = evaluate(neur_a, neur_b, is_text=False)
    print(f"\n{'='*40}")
    print(f"RESULTS:")
    print(f"  Neuralese MSE:  {neur_mse:.6f}  |  Bandwidth: {BOTTLENECK_DIM}×32bit = {compute_bandwidth(False)} bits")
    for label, (ta, tb) in text_models.items():
        mse, _ = evaluate(ta, tb, is_text=True)
        text_mses[label] = mse
        ntoks = text_configs[label]
        print(f"  {label} MSE: {mse:.6f}  |  Bandwidth: {ntoks}×{np.log2(VOCAB_SIZE):.0f}bit = {compute_bandwidth(True, ntoks):.0f} bits")

    # Bandwidth efficiency ratio
    text_best_mse = min(text_mses.values())
    text_best_name = min(text_mses, key=text_mses.get)
    print(f"\n  Neuralese is {text_best_mse/neur_mse:.1f}× more accurate than best text baseline ({text_best_name})")

    # 4. Visualize
    print("\nGenerating visualizations...")
    interp_mse = plot_results(neur_history, text_histories, neur_mse, text_mses, test_x, neur_a)
    print(f"  Interpolation test MSE: {interp_mse:.6f}")

    print(f"\nDone! Outputs in output/")
