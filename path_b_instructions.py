"""
Neuralese Path B — Agent Instruction Following
Tests whether 12D Neuralese can encode actionable instructions for code editing.

Observer: sees full task description (file, line, edit type, context, fix) → 12D z
Navigator: sees ONLY 12D z → predicts (file, line, edit type, fix summary)
No RL, no walls, no navigation — pure information throughput.

Metric: can the Navigator correctly identify WHERE and HOW to edit from z alone?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIM = 12
HIDDEN = 128
EMBED_DIM = 32
EPOCHS = 8000
BATCH = 64
LR = 1e-3

# Task vocabulary
MAX_FILES = 30
MAX_EDIT_TYPES = 6        # replace, insert, delete, append, prepend, modify
MAX_FUNCTIONS = 64
LINE_MAX = 500
CONTEXT_LINES = 3          # Lines of context per task
CONTEXT_EMBED = 32         # Embedding per context line
ERROR_EMBED = 32           # Error message embedding
MAX_ERRORS = 20

# Observer input dim
OBSERVER_INPUT = (
    EMBED_DIM               # file path embedding
    + 12                    # line number bits
    + MAX_EDIT_TYPES        # edit type one-hot
    + MAX_FUNCTIONS         # function name one-hot
    + CONTEXT_LINES * CONTEXT_EMBED  # context embeddings
    + ERROR_EMBED           # error message embedding
)  # 32 + 12 + 6 + 64 + 96 + 32 = 242

# Navigator output: (file_id, line_number, edit_type_id)
NAV_OUTPUT_FILE = MAX_FILES
NAV_OUTPUT_LINE = 1        # Scalar
NAV_OUTPUT_EDIT = MAX_EDIT_TYPES
NAV_OUTPUT_TOTAL = NAV_OUTPUT_FILE + NAV_OUTPUT_LINE + NAV_OUTPUT_EDIT


# --- TASK GENERATOR ---

FILES = [f"src/module_{i}.py" for i in range(20)] + [
    f"lib/utils_{i}.py" for i in range(5)] + [
    f"tests/test_{i}.py" for i in range(5)]

EDIT_TYPES = ["replace", "insert", "delete", "append", "prepend", "modify"]

FUNCTIONS = [
    "evaluate", "train", "forward", "backward", "optimize",
    "load_config", "save_model", "process_batch", "validate_input",
    "transform", "encode", "decode", "compress", "decompress",
    "read_file", "write_file", "search", "parse_args",
    "build_prompt", "parse_response", "generate", "classify",
    "run_pipeline", "preprocess", "postprocess", "normalize",
    "extract_features", "reduce_dim", "augment", "validate",
    "handle_error", "format_output", "sanitize_input",
    "cache_result", "log_metrics", "check_bounds",
    "apply_patch", "compute_hash", "verify_signature",
    "serialize", "deserialize", "merge_configs",
    "detect_anomaly", "filter_noise", "interpolate",
    "extrapolate", "convolve", "deconvolve",
    "quantize", "threshold", "smooth",
    "align_sequences", "cluster_data", "rank_items",
    "shuffle", "split_dataset", "cross_validate",
    "compute_gradient", "apply_dropout", "batch_norm",
    "layer_norm", "attention", "embed",
] * 2  # Pad to 128, take first MAX_FUNCTIONS
FUNCTIONS = list(dict.fromkeys(FUNCTIONS))[:MAX_FUNCTIONS]

ERRORS = [
    "TypeError: expected str, got int",
    "IndexError: list index out of range",
    "KeyError: 'config' not found",
    "ValueError: invalid literal for int()",
    "AttributeError: 'NoneType' object has no attribute 'forward'",
    "RuntimeError: tensor shape mismatch",
    "ImportError: No module named 'utils'",
    "OSError: file not found",
    "AssertionError: x > 0 failed",
    "TimeoutError: operation timed out",
    "ConnectionError: refused",
    "MemoryError: allocation failed",
    "SyntaxError: invalid syntax",
    "NameError: name 'x' is not defined",
    "ZeroDivisionError: division by zero",
    "FileNotFoundError: config.yaml",
    "PermissionError: access denied",
    "NotImplementedError: abstract method",
    "RecursionError: maximum depth exceeded",
    "OverflowError: math range error",
]

CONTEXT_POOL = [
    "def process(data, config=None):",
    "    result = self.forward(data)",
    "    if result is None:",
    "    with open(path, 'r') as f:",
    "    model = load_model(checkpoint)",
    "    x = torch.tensor(values)",
    "    for i in range(len(items)):",
    "    try:",
    "    except Exception as e:",
    "    return self.cache.get(key)",
    "    self.history.append(event)",
    "    logger.info(f'Processing {name}')",
    "    optimizer.step()",
    "    loss = criterion(output, target)",
    "    accuracy = (pred == label).mean()",
    "    embeddings = self.embed(tokens)",
    "    attention_weights = F.softmax(scores)",
    "    hidden = self.gru(embedded, hidden)",
    "    output = self.classifier(features)",
    "    mask = (input_ids != pad_token_id)",
]


def _hash_str(s: str, dim: int) -> torch.Tensor:
    if not s: return torch.zeros(dim)
    h = hashlib.sha256(s.encode()).digest()
    bits = []
    for byte in h:
        for bit_pos in range(8):
            bits.append(float((byte >> bit_pos) & 1))
    return torch.tensor(bits[:dim] + [0.0]*(dim-len(bits)) if len(bits)<dim else bits[:dim], dtype=torch.float32)


class TaskBank:
    def __init__(self, size=1000):
        self.tasks = [self._generate() for _ in range(size)]

    def _generate(self):
        file = np.random.choice(FILES)
        line = np.random.randint(1, LINE_MAX)
        edit = np.random.choice(EDIT_TYPES)
        fn = np.random.choice(FUNCTIONS)
        err = np.random.choice(ERRORS)
        ctx_before = [np.random.choice(CONTEXT_POOL) for _ in range(CONTEXT_LINES)]
        ctx_after = [np.random.choice(CONTEXT_POOL) for _ in range(CONTEXT_LINES)]
        return (file, line, edit, fn, ctx_before, ctx_after, err)

    def sample(self, n):
        idx = np.random.choice(len(self.tasks), n, replace=False)
        return [self.tasks[i] for i in idx]

    def to_observer_input(self, tasks, file_embed, fn_embed, ctx_embed, err_embed):
        """Convert list of tasks to Observer input tensor [B, 242]."""
        B = len(tasks)
        x = torch.zeros(B, OBSERVER_INPUT)

        for b, (file, line, edit, fn, ctx_before, ctx_after, err) in enumerate(tasks):
            offset = 0

            # File path → embedding
            fid = FILES.index(file)
            x[b, offset:offset+EMBED_DIM] = file_embed(torch.tensor(fid))
            offset += EMBED_DIM

            # Line number → binary bits
            for i in range(12):
                if line & (1 << i): x[b, offset+i] = 1.0
            offset += 12

            # Edit type → one-hot
            eid = EDIT_TYPES.index(edit)
            x[b, offset+eid] = 1.0
            offset += MAX_EDIT_TYPES

            # Function name → one-hot
            fnid = FUNCTIONS.index(fn)
            x[b, offset+fnid] = 1.0
            offset += MAX_FUNCTIONS

            # Context lines → embeddings (hash-based for now)
            for i, ctx in enumerate(ctx_before):
                h = _hash_str(ctx, CONTEXT_EMBED)
                x[b, offset+i*CONTEXT_EMBED:offset+(i+1)*CONTEXT_EMBED] = h
            offset += CONTEXT_LINES * CONTEXT_EMBED

            # Error → embedding
            eid = ERRORS.index(err)
            x[b, offset:offset+ERROR_EMBED] = err_embed(torch.tensor(eid))

        return x

    def navigator_targets(self, tasks):
        """Convert tasks to Navigator targets: [B, file_id], [B], [B, edit_id]."""
        B = len(tasks)
        files = torch.zeros(B, dtype=torch.long)
        lines = torch.zeros(B, dtype=torch.float32)
        edits = torch.zeros(B, dtype=torch.long)

        for b, (file, line, edit, _, _, _, _) in enumerate(tasks):
            files[b] = FILES.index(file)
            lines[b] = line / LINE_MAX  # Normalize to [0, 1]
            edits[b] = EDIT_TYPES.index(edit)

        return files, lines, edits


# --- MODELS ---

class Observer(nn.Module):
    """Encodes task description → 12D Neuralese instruction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVER_INPUT, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class Navigator(nn.Module):
    """12D Neuralese → (file prediction, line prediction, edit type prediction)."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.file_head = nn.Linear(HIDDEN, NAV_OUTPUT_FILE)
        self.line_head = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())
        self.edit_head = nn.Linear(HIDDEN, NAV_OUTPUT_EDIT)

    def forward(self, z):
        h = self.shared(z)
        return self.file_head(h), self.line_head(h).squeeze(-1), self.edit_head(h)


# --- TRAINING ---

def train(observer, navigator, task_bank,
          file_embed, fn_embed, ctx_embed, err_embed,
          epochs=EPOCHS):
    params = (list(observer.parameters()) + list(navigator.parameters())
              + list(file_embed.parameters()) + list(fn_embed.parameters())
              + list(ctx_embed.parameters()) + list(err_embed.parameters()))
    opt = optim.Adam(params, lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"loss": [], "file_acc": [], "line_mae": [], "edit_acc": []}

    for ep in range(epochs):
        tasks = task_bank.sample(BATCH)
        x = task_bank.to_observer_input(tasks, file_embed, fn_embed, ctx_embed, err_embed)
        target_file, target_line, target_edit = task_bank.navigator_targets(tasks)

        z = observer(x)
        pred_file, pred_line, pred_edit = navigator(z)

        loss_file = nn.CrossEntropyLoss()(pred_file, target_file)
        loss_line = nn.MSELoss()(pred_line, target_line)
        loss_edit = nn.CrossEntropyLoss()(pred_edit, target_edit)
        loss = loss_file + loss_line + loss_edit

        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

        hist["loss"].append(loss.item())
        hist["file_acc"].append((pred_file.argmax(-1) == target_file).float().mean().item())
        hist["line_mae"].append((pred_line - target_line).abs().mean().item())
        hist["edit_acc"].append((pred_edit.argmax(-1) == target_edit).float().mean().item())

        if ep % 2000 == 0:
            print(f"  Step {ep:5d}: loss={loss.item():.4f} "
                  f"file={hist['file_acc'][-1]:.1%} "
                  f"line_mae={hist['line_mae'][-1]:.3f} "
                  f"edit={hist['edit_acc'][-1]:.1%}")

    return hist


def evaluate(observer, navigator, task_bank,
             file_embed, fn_embed, ctx_embed, err_embed,
             n=200):
    tasks = task_bank.sample(n)
    x = task_bank.to_observer_input(tasks, file_embed, fn_embed, ctx_embed, err_embed)
    target_file, target_line, target_edit = task_bank.navigator_targets(tasks)

    with torch.no_grad():
        z = observer(x)
        pred_file, pred_line, pred_edit = navigator(z)

    file_acc = (pred_file.argmax(-1) == target_file).float().mean().item()
    line_mae = (pred_line - target_line).abs().mean().item()
    edit_acc = (pred_edit.argmax(-1) == target_edit).float().mean().item()

    # Exact match: all three correct
    file_ok = pred_file.argmax(-1) == target_file
    line_ok = (pred_line - target_line).abs() < 0.02  # Within ~10 lines
    edit_ok = pred_edit.argmax(-1) == target_edit
    exact_match = (file_ok & line_ok & edit_ok).float().mean().item()

    return {
        "file_accuracy": file_acc,
        "line_mae": line_mae,
        "line_mae_abs": line_mae * LINE_MAX,  # In actual line numbers
        "edit_accuracy": edit_acc,
        "exact_match": exact_match,
    }


def plot_results(hist, eval_r, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(hist["loss"], alpha=0.3, color='blue')
    ax.set_title("Training Loss"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(hist["file_acc"], alpha=0.6, label="File accuracy", color='green')
    ax.plot(hist["edit_acc"], alpha=0.6, label="Edit accuracy", color='orange')
    ax.set_title("Categorical Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    ax.plot(hist["line_mae"], alpha=0.6, color='red')
    ax.set_title(f"Line Number MAE (normalized)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    metrics = ["File\nAccuracy", "Line\nMAE", "Edit\nAccuracy", "Exact\nMatch"]
    vals = [eval_r["file_accuracy"], eval_r["line_mae_abs"],
            eval_r["edit_accuracy"], eval_r["exact_match"]]
    colors = ['green', 'red', 'orange', 'blue']
    bars = ax.bar(metrics, vals, color=colors, alpha=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.3f}" if val < 1 else f"{val:.0f}", ha='center')
    ax.set_title("Evaluation")
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Neuralese Path B — Agent Instruction Following ({LATENT_DIM}D bottleneck)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir/"path_b_results.png", dpi=150)
    plt.close()


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"NEURALESE PATH B — Agent Instruction Following")
    print("=" * 60)
    print(f"  Bottleneck: {LATENT_DIM}D")
    print(f"  Observer input: {OBSERVER_INPUT}D (file+line+edit+context+error)")
    print(f"  Navigator output: file({MAX_FILES}) + line(1) + edit({MAX_EDIT_TYPES})")
    print(f"  Task bank: synthetic code-editing tasks")

    # Learnable embeddings
    file_embed = nn.Embedding(MAX_FILES, EMBED_DIM)
    fn_embed = nn.Embedding(MAX_FUNCTIONS, MAX_FUNCTIONS)  # one-hot pass-through
    ctx_embed = nn.Embedding(len(CONTEXT_POOL), CONTEXT_EMBED)
    err_embed = nn.Embedding(MAX_ERRORS, ERROR_EMBED)

    task_bank = TaskBank(size=2000)

    print("\n[Training]...")
    observer = Observer()
    navigator = Navigator()
    hist = train(observer, navigator, task_bank,
                 file_embed, fn_embed, ctx_embed, err_embed)

    print("\n[Evaluation]...")
    eval_r = evaluate(observer, navigator, task_bank,
                      file_embed, fn_embed, ctx_embed, err_embed, n=200)
    print(f"  File accuracy:      {eval_r['file_accuracy']:.1%}")
    print(f"  Line MAE (absolute): {eval_r['line_mae_abs']:.1f} lines")
    print(f"  Edit type accuracy:  {eval_r['edit_accuracy']:.1%}")
    print(f"  Exact match:         {eval_r['exact_match']:.1%}")

    # Bandwidth comparison
    bits_per_z = LATENT_DIM * 32
    text_tokens = 80  # Approximate tokens needed to describe a task
    bits_per_text = text_tokens * 16.6
    print(f"\n  Bandwidth: {bits_per_z} bits (Neuralese) vs ~{bits_per_text:.0f} bits (text)")
    print(f"  Efficiency: {bits_per_text/bits_per_z:.0f}x")

    print("\nGenerating plots...")
    plot_results(hist, eval_r, out_dir)
    print(f"\nDone! Outputs in {out_dir}/")
