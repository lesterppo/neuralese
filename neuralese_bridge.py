"""
Neuralese Bridge — Structured Context Compression for Agent Subcalls
Tier 2: Replaces text-based context strings in delegate_task with Neuralese vectors.

Pattern:
  Parent agent:  encode(structured_context) → 16D Neuralese vector
  Subagent:      decode(neuralese_vector) → structured_context → injected into system prompt

Token savings: 300-800 token context string → 16 floats (zero tokenization cost)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
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

# Structured context schema
MAX_FILE_PATH_HASH = 64    # Hash file paths to fixed-size binary vectors
MAX_ERROR_TYPES = 32       # Categorical error types
MAX_FUNCTION_NAMES = 128   # Function name vocabulary
MAX_ATTEMPTED_FIXES = 3    # Max number of attempted fixes
FIX_DESC_DIM = 64          # Hash each fix description to 64D
MAX_SYMBOLS = 5            # Max related symbols
SYMBOL_DIM = 64            # Hash each symbol to 64D
LINE_NUMBER_BITS = 12      # Encode line number in 12 bits (0-4095)
WORKSPACE_HASH_DIM = 64    # Hash workspace path to 64D

TOTAL_INPUT_DIM = (
    MAX_FILE_PATH_HASH      # file path hash (64)
    + LINE_NUMBER_BITS       # line number bits (12)
    + MAX_ERROR_TYPES        # error type one-hot (32)
    + MAX_FUNCTION_NAMES     # function name one-hot (128)
    + MAX_ATTEMPTED_FIXES * FIX_DESC_DIM  # attempted fixes (3*64=192)
    + MAX_SYMBOLS * SYMBOL_DIM             # related symbols (5*64=320)
    + WORKSPACE_HASH_DIM    # workspace path hash (64)
)  # Total: 64+12+32+128+192+320+64 = 812


@dataclass
class StructuredContext:
    """Structured representation of a subagent call context."""
    file_path: str = ""
    line_number: int = 0
    error_type: str = ""           # e.g., "type_error", "import_error", "value_error"
    function_name: str = ""
    attempted_fixes: List[str] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    workspace_path: str = ""

    # Pre-defined error type vocabulary
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

    # Pre-defined function name vocabulary (common in Hermes subagent calls)
    FUNCTION_NAMES = [
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
        "extract_features", "reduce_dim", "augment", "normalize",
    ] * 3  # Pad to 128 by repeating

    def to_vector(self) -> torch.Tensor:
        """Convert structured context to fixed-size input vector."""
        vec = torch.zeros(TOTAL_INPUT_DIM)
        offset = 0

        # File path → hash
        path_hash = self._hash_str(self.file_path, MAX_FILE_PATH_HASH)
        vec[offset:offset + MAX_FILE_PATH_HASH] = path_hash
        offset += MAX_FILE_PATH_HASH

        # Line number → binary bits
        line_bits = self._int_to_bits(self.line_number, LINE_NUMBER_BITS)
        vec[offset:offset + LINE_NUMBER_BITS] = line_bits
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

        # Attempted fixes → hashed vectors
        for i in range(MAX_ATTEMPTED_FIXES):
            if i < len(self.attempted_fixes):
                fix_hash = self._hash_str(self.attempted_fixes[i], FIX_DESC_DIM)
                vec[offset + i * FIX_DESC_DIM:offset + (i + 1) * FIX_DESC_DIM] = fix_hash
            # else: leave as zeros
        offset += MAX_ATTEMPTED_FIXES * FIX_DESC_DIM

        # Related symbols → hashed vectors
        for i in range(MAX_SYMBOLS):
            if i < len(self.related_symbols):
                sym_hash = self._hash_str(self.related_symbols[i], SYMBOL_DIM)
                vec[offset + i * SYMBOL_DIM:offset + (i + 1) * SYMBOL_DIM] = sym_hash
            # else: leave as zeros
        offset += MAX_SYMBOLS * SYMBOL_DIM

        # Workspace path → hash
        ws_hash = self._hash_str(self.workspace_path, WORKSPACE_HASH_DIM)
        vec[offset:offset + WORKSPACE_HASH_DIM] = ws_hash

        return vec

    def to_text_context(self) -> str:
        """Convert back to human-readable text (for subagent system prompt)."""
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
    def _hash_str(s: str, dim: int) -> torch.Tensor:
        """Hash a string to a fixed-size binary vector using SHA256."""
        if not s:
            return torch.zeros(dim)
        h = hashlib.sha256(s.encode()).digest()
        bits = []
        for byte in h:
            for bit_pos in range(8):
                bits.append(float((byte >> bit_pos) & 1))
        bits = bits[:dim]
        if len(bits) < dim:
            bits.extend([0.0] * (dim - len(bits)))
        return torch.tensor(bits, dtype=torch.float32)

    @staticmethod
    def _int_to_bits(n: int, num_bits: int) -> torch.Tensor:
        """Convert integer to binary bit vector."""
        bits = []
        for i in range(num_bits):
            bits.append(float((n >> i) & 1))
        return torch.tensor(bits, dtype=torch.float32)

    @staticmethod
    def _safe_index(item: str, vocab: List[str]) -> int:
        """Get index in vocabulary, or len(vocab) if not found."""
        try:
            return vocab.index(item)
        except ValueError:
            return len(vocab)


# --- MODELS ---

class ContextEncoder(nn.Module):
    """Encodes structured context (812D) → Neuralese vector (16D)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOTAL_INPUT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class ContextDecoder(nn.Module):
    """Decodes Neuralese vector (16D) → structured context reconstruction (812D)."""
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


# --- SYNTHETIC DATA GENERATION ---

ERROR_TYPES = StructuredContext.ERROR_TYPES[:MAX_ERROR_TYPES]
FUNCTION_NAMES = list(dict.fromkeys(StructuredContext.FUNCTION_NAMES))[:MAX_FUNCTION_NAMES]

SAMPLE_PATHS = [
    "/home/peter/neuralese/maze_navigator_v5.py",
    "/home/peter/neuralese/maze_navigator_v4.py",
    "/home/peter/hermes-agent/tools/delegate_tool.py",
    "/home/peter/hermes-agent/run_agent.py",
    "/home/peter/hermes-agent/model_tools.py",
    "/home/peter/neuralese/demo.py",
    "/home/peter/ply-tensor-language/ply/interpreter.py",
    "/home/peter/deepworld/src/world.py",
    "/home/peter/stock-world-model/dual_momentum.py",
    "/home/peter/hermes-agent/cli.py",
]

SAMPLE_SYMBOLS = [
    "navigator", "observer", "evaluate", "forward",
    "delegate_task", "chat", "run_conversation", "parse_args",
    "load_config", "save_output", "validate_input", "process_batch",
    "ppo_update", "compute_gae", "warm_start", "collect_trajectories",
]

SAMPLE_FIXES = [
    "added .squeeze(0) to fix tensor shape",
    "replaced shell=True with subprocess.run(input=...)",
    "fixed NoneType error with null check",
    "updated function signature to match new API",
    "added type cast for tensor conversion",
    "patched path resolution for WSL compatibility",
    "added timeout handler for long-running operations",
    "fixed index out of bounds with bounds check",
    "replaced deprecated API call",
    "added error handling for missing environment variable",
    "fixed race condition with lock",
]


def generate_synthetic_context() -> StructuredContext:
    """Generate a random synthetic subagent call context."""
    return StructuredContext(
        file_path=np.random.choice(SAMPLE_PATHS),
        line_number=np.random.randint(1, 2000),
        error_type=np.random.choice(ERROR_TYPES),
        function_name=np.random.choice(FUNCTION_NAMES),
        attempted_fixes=list(np.random.choice(
            SAMPLE_FIXES,
            size=np.random.randint(0, min(4, len(SAMPLE_FIXES))),
            replace=False,
        )),
        related_symbols=list(np.random.choice(
            SAMPLE_SYMBOLS,
            size=np.random.randint(1, min(6, len(SAMPLE_SYMBOLS))),
            replace=False,
        )),
        workspace_path=np.random.choice(SAMPLE_PATHS[:5]),
    )


def generate_batch(batch_size: int) -> torch.Tensor:
    """Generate batch of context vectors."""
    contexts = [generate_synthetic_context() for _ in range(batch_size)]
    return torch.stack([c.to_vector() for c in contexts]), contexts


# --- TRAINING ---

def train(encoder, decoder, epochs=EPOCHS):
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []

    for step in range(epochs):
        x, _ = generate_batch(BATCH)
        z = encoder(x)
        x_hat = decoder(z)

        task_loss = nn.MSELoss()(x_hat, x)
        l2_loss = L2_LAMBDA * torch.norm(z, p=2)
        loss = task_loss + l2_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history.append(loss.item())

        if step % 2000 == 0:
            print(f"  Step {step:5d}: loss={loss.item():.6f}  task={task_loss.item():.6f}")

    return history


# --- EVALUATION ---

def evaluate_roundtrip(encoder, decoder, num_samples=100):
    """Measure how well the Neuralese vector preserves structured context."""
    mse_total = 0.0
    exact_matches = {"error_type": 0, "function_name": 0}

    for _ in range(num_samples):
        ctx = generate_synthetic_context()
        x = ctx.to_vector().unsqueeze(0)

        with torch.no_grad():
            z = encoder(x)
            x_hat = decoder(z).squeeze(0)

        mse_total += nn.MSELoss()(x_hat, x).item()

        # Decode error type from one-hot region
        err_start = MAX_FILE_PATH_HASH + LINE_NUMBER_BITS
        err_region = x_hat[err_start:err_start + MAX_ERROR_TYPES]
        err_pred = torch.argmax(err_region).item()
        err_true = ERROR_TYPES.index(ctx.error_type) if ctx.error_type in ERROR_TYPES else -1
        if err_pred == err_true:
            exact_matches["error_type"] += 1

        # Decode function name
        fn_start = err_start + MAX_ERROR_TYPES
        fn_region = x_hat[fn_start:fn_start + MAX_FUNCTION_NAMES]
        fn_pred = torch.argmax(fn_region).item()
        fn_true = FUNCTION_NAMES.index(ctx.function_name) if ctx.function_name in FUNCTION_NAMES else -1
        if fn_pred == fn_true:
            exact_matches["function_name"] += 1

    return {
        "mse": mse_total / num_samples,
        "error_type_accuracy": exact_matches["error_type"] / num_samples,
        "function_accuracy": exact_matches["function_name"] / num_samples,
    }


def demo_compression(encoder, decoder):
    """Demonstrate a real compression/decompression cycle."""
    ctx = StructuredContext(
        file_path="/home/peter/neuralese/maze_navigator_v5.py",
        line_number=415,
        error_type="type_error",
        function_name="evaluate",
        attempted_fixes=["added .squeeze(0) at line 414", "checked tensor dimensions"],
        related_symbols=["navigator", "observer", "forward"],
        workspace_path="/home/peter/neuralese",
    )

    x = ctx.to_vector().unsqueeze(0)
    with torch.no_grad():
        z = encoder(x)
        x_hat = decoder(z).squeeze(0)

    # Reconstruct error type
    err_start = MAX_FILE_PATH_HASH + LINE_NUMBER_BITS
    err_region = x_hat[err_start:err_start + MAX_ERROR_TYPES]
    err_pred_idx = torch.argmax(err_region).item()
    err_pred = ERROR_TYPES[err_pred_idx] if err_pred_idx < len(ERROR_TYPES) else "unknown"

    # Reconstruct function name
    fn_start = err_start + MAX_ERROR_TYPES
    fn_region = x_hat[fn_start:fn_start + MAX_FUNCTION_NAMES]
    fn_pred_idx = torch.argmax(fn_region).item()
    fn_pred = FUNCTION_NAMES[fn_pred_idx] if fn_pred_idx < len(FUNCTION_NAMES) else "unknown"

    # Reconstruct line number
    ln_start = MAX_FILE_PATH_HASH
    line_bits = x_hat[ln_start:ln_start + LINE_NUMBER_BITS]
    line_pred = int(sum(b.item() > 0.5 for b in line_bits))  # Quick decode

    mse = nn.MSELoss()(x_hat, x).item()

    return {
        "original": ctx.to_text_context(),
        "reconstructed_error": err_pred,
        "original_error": ctx.error_type,
        "reconstructed_function": fn_pred,
        "original_function": ctx.function_name,
        "reconstructed_line": line_pred,
        "original_line": ctx.line_number,
        "mse": mse,
        "latent_vector": z.squeeze(0).tolist(),
        "latent_dim": LATENT_DIM,
    }


# --- VISUALIZATION ---

def plot_results(history, eval_results, demo_result, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history, alpha=0.5, color='blue')
    ax.set_title("Context Encoder Training")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    metrics = ["MSE", "Error Type\nAccuracy", "Function\nAccuracy"]
    vals = [eval_results["mse"], eval_results["error_type_accuracy"],
            eval_results["function_accuracy"]]
    colors = ['red', 'green', 'blue']
    bars = ax.bar(metrics, vals, color=colors, alpha=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontweight='bold')
    ax.set_title("Round-Trip Evaluation")
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    latent = demo_result["latent_vector"]
    ax.bar(range(len(latent)), latent, color='purple', alpha=0.6)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f"Neuralese Vector ({LATENT_DIM}D)")
    ax.set_xlabel("Dimension"); ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    demo_text = (
        f"Original context:\n{demo_result['original']}\n\n"
        f"Reconstructed:\n"
        f"  Error: {demo_result['reconstructed_error']} "
        f"(was: {demo_result['original_error']})\n"
        f"  Function: {demo_result['reconstructed_function']} "
        f"(was: {demo_result['original_function']})\n"
        f"  Line: ~{demo_result['reconstructed_line']} "
        f"(was: {demo_result['original_line']})\n"
        f"\nMSE: {demo_result['mse']:.6f}\n"
        f"Compression: {LATENT_DIM} floats vs ~300 text tokens"
    )
    ax.text(0.05, 0.5, demo_text, fontfamily='monospace', fontsize=8, va='center',
            transform=ax.transAxes)
    ax.set_title("Demo Round-Trip"); ax.axis('off')

    plt.suptitle("Neuralese Bridge — Context Compression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = out_dir / "bridge_results.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NEURALESE BRIDGE — Context Compression for Agent Subcalls")
    print("=" * 60)
    print(f"  Latent dim: {LATENT_DIM}")
    print(f"  Input dim: {TOTAL_INPUT_DIM}")
    print(f"  Compression ratio: {TOTAL_INPUT_DIM / LATENT_DIM:.1f}×")

    print("\n[Training] Context Encoder/Decoder...")
    encoder = ContextEncoder()
    decoder = ContextDecoder()
    history = train(encoder, decoder)

    print("\n[Evaluation] Round-trip accuracy...")
    eval_results = evaluate_roundtrip(encoder, decoder)
    print(f"  MSE: {eval_results['mse']:.6f}")
    print(f"  Error type accuracy: {eval_results['error_type_accuracy']:.1%}")
    print(f"  Function name accuracy: {eval_results['function_accuracy']:.1%}")

    print("\n[Demo] Single context compression...")
    demo_result = demo_compression(encoder, decoder)
    print(f"  Original error: {demo_result['original_error']}")
    print(f"  Reconstructed:  {demo_result['reconstructed_error']}")
    print(f"  Original function: {demo_result['original_function']}")
    print(f"  Reconstructed:     {demo_result['reconstructed_function']}")
    print(f"  MSE: {demo_result['mse']:.6f}")

    print("\nGenerating plots...")
    plot_results(history, eval_results, demo_result, out_dir)

    # Save model
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "config": {"latent_dim": LATENT_DIM, "input_dim": TOTAL_INPUT_DIM},
    }, out_dir / "bridge_model.pt")
    print(f"  Model saved: {out_dir / 'bridge_model.pt'}")

    print(f"\nDone! Outputs in {out_dir}/")
