"""
Neuralese Bridge v2 — Learnable Embeddings for Structured Context
Replaces SHA256 hashing with PyTorch Embedding layers.
Gemini Flash review: SHA256 creates non-smooth distributions that MLPs can't decode.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIG ---
HIDDEN = 128
LATENT_DIM = 16
EPOCHS = 10000
BATCH = 64
LR = 1e-3
L2_LAMBDA = 0.001

# Structured context schema — learnable embeddings
EMBED_DIM = 32
MAX_FILE_PATHS = 50
MAX_ERROR_TYPES = 32
MAX_FUNCTION_NAMES = 128
MAX_ATTEMPTED_FIXES = 3
FIX_EMBED_DIM = 32
MAX_SYMBOLS = 5
SYMBOL_EMBED_DIM = 32
LINE_NUMBER_BITS = 12
WORKSPACE_EMBED_DIM = 32
MAX_WORKSPACES = 20

TOTAL_INPUT_DIM = (
    EMBED_DIM + LINE_NUMBER_BITS + MAX_ERROR_TYPES + MAX_FUNCTION_NAMES
    + MAX_ATTEMPTED_FIXES * FIX_EMBED_DIM + MAX_SYMBOLS * SYMBOL_EMBED_DIM
    + WORKSPACE_EMBED_DIM
)  # 32+12+32+128+96+160+32 = 492


@dataclass
class StructuredContext:
    file_path: str = ""
    line_number: int = 0
    error_type: str = ""           # e.g., "type_error", "import_error"
    function_name: str = ""
    attempted_fixes: List[str] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    workspace_path: str = ""

    ERROR_TYPES = [
        "type_error", "import_error", "value_error", "attribute_error",
        "key_error", "index_error", "name_error", "syntax_error",
        "runtime_error", "os_error", "io_error", "assertion_error",
        "timeout_error", "memory_error", "connection_error", "permission_error",
        "not_implemented", "deprecation_warning", "resource_exhausted",
        "integration_error", "config_error", "auth_error", "api_error",
        "data_error", "logic_error", "race_condition", "deadlock",
        "null_reference", "overflow_error", "underflow_error",
        "encoding_error", "serialization_error",
    ]

    FUNCTION_NAMES = (list(dict.fromkeys([
        "evaluate", "train", "warm_start", "rl_fine_tune", "ppo_update",
        "collect_trajectories", "compute_gae", "latent_evolution_test",
        "gen_maze", "bfs_reachable", "a_star_path", "get_radar",
        "forward", "backward", "step", "optimize",
        "load_config", "save_model", "process_batch", "run_pipeline",
        "validate", "transform", "preprocess", "postprocess",
        "encode", "decode", "compress", "decompress",
        "read_file", "write_file", "search_files", "patch",
        "terminal", "execute_code", "delegate_task", "handle_function_call",
        "chat", "run_conversation", "build_prompt", "parse_response",
        "generate", "classify", "cluster", "predict",
    ])) * 3)[:MAX_FUNCTION_NAMES]

    def to_vector(self) -> torch.Tensor:
        """Convert to embedding indices (NOT hashed). Actual embedding happens in the model."""
        vec = torch.zeros(TOTAL_INPUT_DIM)
        offset = 0

        # File path — leave zeros, handled by embedding layer
        offset += EMBED_DIM

        # Line number → binary bits
        for i in range(LINE_NUMBER_BITS):
            if self.line_number & (1 << i):
                vec[offset + i] = 1.0
        offset += LINE_NUMBER_BITS

        # Error type → one-hot
        err_idx = self._safe_index(self.error_type, self.ERROR_TYPES)
        if err_idx < MAX_ERROR_TYPES:
            vec[offset + err_idx] = 1.0
        offset += MAX_ERROR_TYPES

        # Function name → one-hot
        fn_idx = self._safe_index(self.function_name, self.FUNCTION_NAMES)
        if fn_idx < MAX_FUNCTION_NAMES:
            vec[offset + fn_idx] = 1.0
        offset += MAX_FUNCTION_NAMES

        # Attempted fixes, related symbols, workspace — zeros (handled by embeddings)
        offset += MAX_ATTEMPTED_FIXES * FIX_EMBED_DIM
        offset += MAX_SYMBOLS * SYMBOL_EMBED_DIM
        offset += WORKSPACE_EMBED_DIM

        return vec

    def to_text_context(self) -> str:
        parts = []
        if self.file_path:
            parts.append(f"File: {self.file_path}, line {self.line_number}")
        if self.error_type:
            parts.append(f"Error: {self.error_type}")
        if self.function_name:
            parts.append(f"Function: {self.function_name}")
        if self.attempted_fixes:
            parts.append(f"Attempted fixes: {'; '.join(self.attempted_fixes)}")
        if self.related_symbols:
            parts.append(f"Related: {', '.join(self.related_symbols)}")
        if self.workspace_path:
            parts.append(f"Workspace: {self.workspace_path}")
        return "\n".join(parts)

    @staticmethod
    def _safe_index(item: str, vocab: List[str]) -> int:
        try: return vocab.index(item)
        except ValueError: return len(vocab)


# --- MODELS ---

class ContextEncoder(nn.Module):
    """Encodes structured context → 16D Neuralese using learnable embeddings."""
    def __init__(self):
        super().__init__()
        # Learnable embeddings for string fields (replaces SHA256 hashing)
        self.path_embed = nn.Embedding(MAX_FILE_PATHS, EMBED_DIM)
        self.fix_embed = nn.Embedding(MAX_ATTEMPTED_FIXES * 20, FIX_EMBED_DIM)  # 60 fix descriptions
        self.symbol_embed = nn.Embedding(MAX_SYMBOLS * 20, SYMBOL_EMBED_DIM)    # 100 symbols
        self.workspace_embed = nn.Embedding(MAX_WORKSPACES, WORKSPACE_EMBED_DIM)

        # MLP encoder
        self.net = nn.Sequential(
            nn.Linear(TOTAL_INPUT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def embed_context(self, contexts: List[StructuredContext]) -> torch.Tensor:
        """Convert a list of StructuredContext to embedding-augmented vectors."""
        B = len(contexts)
        # Build base vectors (one-hot + binary)
        base = torch.stack([c.to_vector() for c in contexts])

        # Build vocabulary indices for embedding fields
        path_ids = torch.zeros(B, dtype=torch.long)
        fix_ids = torch.zeros(B, MAX_ATTEMPTED_FIXES, dtype=torch.long)
        symbol_ids = torch.zeros(B, MAX_SYMBOLS, dtype=torch.long)
        ws_ids = torch.zeros(B, dtype=torch.long)

        # Build simple string→id mappings (hash-based for now, but embeddings are learnable)
        path_vocab = {}
        fix_vocab = {}
        sym_vocab = {}
        ws_vocab = {}

        for b, ctx in enumerate(contexts):
            # File path
            if ctx.file_path not in path_vocab:
                path_vocab[ctx.file_path] = len(path_vocab) % MAX_FILE_PATHS
            path_ids[b] = path_vocab[ctx.file_path]

            # Fix descriptions
            for i, fix in enumerate(ctx.attempted_fixes[:MAX_ATTEMPTED_FIXES]):
                if fix not in fix_vocab:
                    fix_vocab[fix] = len(fix_vocab) % (MAX_ATTEMPTED_FIXES * 20)
                fix_ids[b, i] = fix_vocab[fix]

            # Symbols
            for i, sym in enumerate(ctx.related_symbols[:MAX_SYMBOLS]):
                if sym not in sym_vocab:
                    sym_vocab[sym] = len(sym_vocab) % (MAX_SYMBOLS * 20)
                symbol_ids[b, i] = sym_vocab[sym]

            # Workspace
            if ctx.workspace_path not in ws_vocab:
                ws_vocab[ctx.workspace_path] = len(ws_vocab) % MAX_WORKSPACES
            ws_ids[b] = ws_vocab[ctx.workspace_path]

        # Embed and insert into base vector
        path_emb = self.path_embed(path_ids)                    # [B, 32]
        fix_emb = self.fix_embed(fix_ids).view(B, -1)           # [B, 96]
        sym_emb = self.symbol_embed(symbol_ids).view(B, -1)     # [B, 160]
        ws_emb = self.workspace_embed(ws_ids)                    # [B, 32]

        # Place embeddings at correct offsets
        full = base.clone()
        offset = 0
        full[:, offset:offset + EMBED_DIM] = path_emb; offset += EMBED_DIM
        offset += LINE_NUMBER_BITS  # Binary bits stay
        offset += MAX_ERROR_TYPES   # One-hot stays
        offset += MAX_FUNCTION_NAMES  # One-hot stays
        full[:, offset:offset + MAX_ATTEMPTED_FIXES * FIX_EMBED_DIM] = fix_emb
        offset += MAX_ATTEMPTED_FIXES * FIX_EMBED_DIM
        full[:, offset:offset + MAX_SYMBOLS * SYMBOL_EMBED_DIM] = sym_emb
        offset += MAX_SYMBOLS * SYMBOL_EMBED_DIM
        full[:, offset:offset + WORKSPACE_EMBED_DIM] = ws_emb

        return full

    def forward(self, x):
        return self.net(x)


class ContextDecoder(nn.Module):
    """Decodes 16D Neuralese → structured context reconstruction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, TOTAL_INPUT_DIM),
        )

    def forward(self, z):
        return self.net(z)


# --- SYNTHETIC DATA ---

ERROR_TYPES = StructuredContext.ERROR_TYPES[:MAX_ERROR_TYPES]
FN_NAMES = StructuredContext.FUNCTION_NAMES[:MAX_FUNCTION_NAMES]

SAMPLE_PATHS = [
    "/workspace/neuralese/maze_navigator.py",
    "/workspace/neuralese/demo.py",
    "/workspace/project/tools/delegate_tool.py",
    "/workspace/project/run_agent.py",
    "/workspace/project/model_tools.py",
    "/workspace/project/src/utils.py",
    "/workspace/project/src/world.py",
    "/workspace/project/dual_momentum.py",
]

SAMPLE_SYMBOLS = [
    "navigator", "observer", "evaluate", "forward",
    "delegate_task", "chat", "run_conversation",
    "load_config", "save_output", "validate_input",
    "ppo_update", "compute_gae", "warm_start", "collect_trajectories",
]

SAMPLE_FIXES = [
    "added .squeeze(0) to fix tensor shape",
    "replaced shell=True with subprocess.run",
    "fixed NoneType error with null check",
    "updated function signature to match new API",
    "added type cast for tensor conversion",
    "patched path resolution for WSL compatibility",
    "added timeout handler for long operations",
    "fixed index out of bounds with bounds check",
    "replaced deprecated API call",
    "added error handling for missing env",
]


def generate_synthetic_context() -> StructuredContext:
    return StructuredContext(
        file_path=np.random.choice(SAMPLE_PATHS),
        line_number=np.random.randint(1, 2000),
        error_type=np.random.choice(ERROR_TYPES),
        function_name=np.random.choice(FN_NAMES),
        attempted_fixes=list(np.random.choice(
            SAMPLE_FIXES, size=np.random.randint(0, min(4, len(SAMPLE_FIXES))), replace=False)),
        related_symbols=list(np.random.choice(
            SAMPLE_SYMBOLS, size=np.random.randint(1, min(6, len(SAMPLE_SYMBOLS))), replace=False)),
        workspace_path=np.random.choice(SAMPLE_PATHS[:5]),
    )


def generate_batch(batch_size):
    contexts = [generate_synthetic_context() for _ in range(batch_size)]
    return contexts


# --- TRAINING ---

def train(encoder, decoder, epochs=EPOCHS):
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []

    for step in range(epochs):
        contexts = generate_batch(BATCH)
        x = encoder.embed_context(contexts)
        z = encoder(x)
        x_hat = decoder(z)

        task_loss = nn.MSELoss()(x_hat, x)
        l2_loss = L2_LAMBDA * torch.norm(z, p=2)
        loss = task_loss + l2_loss

        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        history.append(loss.item())

        if step % 2000 == 0:
            print(f"  Step {step:5d}: loss={loss.item():.6f}  task={task_loss.item():.6f}")

    return history


# --- EVALUATION ---

def evaluate_roundtrip(encoder, decoder, num_samples=100):
    mse_total = 0.0
    exact_matches = {"error_type": 0, "function_name": 0}

    for _ in range(num_samples):
        ctx = generate_synthetic_context()
        x = encoder.embed_context([ctx])

        with torch.no_grad():
            z = encoder(x)
            x_hat = decoder(z).squeeze(0)

        mse_total += nn.MSELoss()(x_hat, x.squeeze(0)).item()

        err_start = EMBED_DIM + LINE_NUMBER_BITS
        err_region = x_hat[err_start:err_start + MAX_ERROR_TYPES]
        err_pred = torch.argmax(err_region).item()
        err_true = ERROR_TYPES.index(ctx.error_type) if ctx.error_type in ERROR_TYPES else -1
        if err_pred == err_true: exact_matches["error_type"] += 1

        fn_start = err_start + MAX_ERROR_TYPES
        fn_region = x_hat[fn_start:fn_start + MAX_FUNCTION_NAMES]
        fn_pred = torch.argmax(fn_region).item()
        fn_true = FN_NAMES.index(ctx.function_name) if ctx.function_name in FN_NAMES else -1
        if fn_pred == fn_true: exact_matches["function_name"] += 1

    return {
        "mse": mse_total / num_samples,
        "error_type_accuracy": exact_matches["error_type"] / num_samples,
        "function_accuracy": exact_matches["function_name"] / num_samples,
    }


def demo_compression(encoder, decoder):
    ctx = StructuredContext(
        file_path="/workspace/neuralese/maze_navigator.py",
        line_number=415,
        error_type="type_error",
        function_name="evaluate",
        attempted_fixes=["added .squeeze(0) at line 414", "checked tensor dimensions"],
        related_symbols=["navigator", "observer", "forward"],
        workspace_path="/workspace/neuralese",
    )
    x = encoder.embed_context([ctx])
    with torch.no_grad():
        z = encoder(x); x_hat = decoder(z).squeeze(0)

    err_start = EMBED_DIM + LINE_NUMBER_BITS
    err_pred_idx = torch.argmax(x_hat[err_start:err_start + MAX_ERROR_TYPES]).item()
    fn_start = err_start + MAX_ERROR_TYPES
    fn_pred_idx = torch.argmax(x_hat[fn_start:fn_start + MAX_FUNCTION_NAMES]).item()

    return {
        "original": ctx.to_text_context(),
        "reconstructed_error": ERROR_TYPES[err_pred_idx] if err_pred_idx < len(ERROR_TYPES) else "?",
        "original_error": ctx.error_type,
        "reconstructed_function": FN_NAMES[fn_pred_idx] if fn_pred_idx < len(FN_NAMES) else "?",
        "original_function": ctx.function_name,
        "mse": nn.MSELoss()(x_hat, x.squeeze(0)).item(),
        "latent_vector": z.squeeze(0).tolist(),
        "latent_dim": LATENT_DIM,
    }


# --- PLOTS ---

def plot_results(history, eval_results, demo_result, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history, alpha=0.5, color='blue')
    axes[0, 0].set_title("Context Encoder Training v2 (embeddings)"); axes[0, 0].set_yscale("log"); axes[0, 0].grid(True, alpha=0.3)
    metrics = ["MSE", "Error Type\nAccuracy", "Function\nAccuracy"]
    vals = [eval_results["mse"], eval_results["error_type_accuracy"], eval_results["function_accuracy"]]
    bars = axes[0, 1].bar(metrics, vals, color=['red', 'green', 'blue'], alpha=0.7)
    for bar, val in zip(bars, vals): axes[0, 1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}", ha='center')
    axes[0, 1].set_title("Round-Trip Eval"); axes[0, 1].set_ylim(0, 1.1); axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 0].bar(range(len(demo_result["latent_vector"])), demo_result["latent_vector"], color='purple', alpha=0.6)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5); axes[1, 0].set_title(f"Latent ({LATENT_DIM}D)"); axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 1].text(0.05, 0.5,
                    f"Original: {demo_result['original']}\n\n"
                    f"Error: {demo_result['reconstructed_error']} (was: {demo_result['original_error']})\n"
                    f"Function: {demo_result['reconstructed_function']} (was: {demo_result['original_function']})\n"
                    f"MSE: {demo_result['mse']:.6f}\nCompression: {LATENT_DIM} floats vs text",
                    fontfamily='monospace', fontsize=8, va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    plt.suptitle("Neuralese Bridge v2 — Learnable Embeddings", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/"bridge_v2_results.png", dpi=150); plt.close()


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("NEURALESE BRIDGE v2 — Learnable Embeddings")
    print("=" * 60)
    print(f"  Latent dim: {LATENT_DIM}, Input dim: {TOTAL_INPUT_DIM}")
    print(f"  Embeddings: path({EMBED_DIM}D), fix({FIX_EMBED_DIM}D), sym({SYMBOL_EMBED_DIM}D), ws({WORKSPACE_EMBED_DIM}D)")

    print("\n[Training]...")
    encoder = ContextEncoder(); decoder = ContextDecoder()
    history = train(encoder, decoder)

    print("\n[Evaluation]...")
    eval_results = evaluate_roundtrip(encoder, decoder)
    print(f"  MSE: {eval_results['mse']:.6f}")
    print(f"  Error type accuracy: {eval_results['error_type_accuracy']:.1%}")
    print(f"  Function accuracy: {eval_results['function_accuracy']:.1%}")

    print("\n[Demo]...")
    demo_result = demo_compression(encoder, decoder)
    print(f"  Error: {demo_result['original_error']} → {demo_result['reconstructed_error']}")
    print(f"  Function: {demo_result['original_function']} → {demo_result['reconstructed_function']}")
    print(f"  MSE: {demo_result['mse']:.6f}")

    print("\nGenerating plots...")
    plot_results(history, eval_results, demo_result, out_dir)
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, out_dir/"bridge_v2_model.pt")
    print(f"  Model saved: {out_dir / 'bridge_v2_model.pt'}")
    print(f"\nDone! Outputs in {out_dir}/")
