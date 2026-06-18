"""
Neuralese Cross-Model Integration Test
Tests whether compressed Neuralese context preserves enough information
for an LLM to take the same action as with full text context.

Setup:
  1. DeepSeek API (planner) → generates a structured code-editing task
  2. PyTorch Observer → encodes task into 12D Neuralese
  3. DeepSeek API (executor) → receives decoded context, takes action
  4. Compare: text-only vs Neuralese accuracy

Token efficiency measurement included.
"""
import torch
import torch.nn as nn
import json
import subprocess
import numpy as np
from pathlib import Path

# --- Load trained models from Path B ---
LATENT_DIM = 12; HIDDEN = 128
MAX_FILES = 30; MAX_EDIT_TYPES = 6; MAX_FUNCTIONS = 64
EMBED_DIM = 32; CONTEXT_LINES = 3; CONTEXT_EMBED = 32
ERROR_EMBED = 32; MAX_ERRORS = 20; LINE_MAX = 500
OBSERVER_INPUT = (EMBED_DIM + 12 + MAX_EDIT_TYPES + MAX_FUNCTIONS
                  + CONTEXT_LINES * CONTEXT_EMBED + ERROR_EMBED)

FILES = [f"src/module_{i}.py" for i in range(20)] + [f"lib/utils_{i}.py" for i in range(5)] + [f"tests/test_{i}.py" for i in range(5)]
EDIT_TYPES = ["replace","insert","delete","append","prepend","modify"]
FUNCTIONS = ["evaluate","train","forward","backward","optimize","load_config","save_model",
    "process_batch","validate_input","transform","encode","decode","compress","decompress",
    "read_file","write_file","search","parse_args","build_prompt","parse_response","generate",
    "classify","run_pipeline","preprocess","postprocess","normalize","extract_features",
    "reduce_dim","augment","validate","handle_error","format_output","sanitize_input",
    "cache_result","log_metrics","check_bounds","apply_patch","compute_hash",
    "verify_signature","serialize","deserialize","merge_configs","detect_anomaly",
    "filter_noise","interpolate","extrapolate","convolve","deconvolve","quantize",
    "threshold","smooth","align_sequences","cluster_data","rank_items","shuffle",
    "split_dataset","cross_validate","compute_gradient","apply_dropout","batch_norm",
    "layer_norm","attention","embed"][:MAX_FUNCTIONS]

ERRORS = ["TypeError: expected str, got int","IndexError: list index out of range",
    "KeyError: config not found","ValueError: invalid literal","AttributeError: NoneType",
    "RuntimeError: tensor shape mismatch","ImportError: No module named utils",
    "OSError: file not found","AssertionError: x > 0 failed","TimeoutError: operation timed out",
    "ConnectionError: refused","MemoryError: allocation failed","SyntaxError: invalid syntax",
    "NameError: name x not defined","ZeroDivisionError: division by zero",
    "FileNotFoundError: config.yaml","PermissionError: access denied",
    "NotImplementedError: abstract method","RecursionError: maximum depth","OverflowError: math range error"]

import path_b_instructions as pb
sys_path = str(Path(__file__).parent)
import sys; sys.path.insert(0, sys_path)

# --- Load Neuralese models ---

class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVER_INPUT,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),
            nn.Linear(HIDDEN,HIDDEN//2),nn.ReLU(),nn.Linear(HIDDEN//2,LATENT_DIM),nn.LayerNorm(LATENT_DIM))
    def forward(self,x): return self.net(x)

class Navigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(LATENT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
        self.file_head=nn.Linear(HIDDEN,MAX_FILES); self.line_head=nn.Sequential(nn.Linear(HIDDEN,1),nn.Sigmoid())
        self.edit_head=nn.Linear(HIDDEN,MAX_EDIT_TYPES)
    def forward(self,z):
        h=self.shared(z); return self.file_head(h),self.line_head(h).squeeze(-1),self.edit_head(h)


# Quick retrain (Path B already proven, just need a trained instance)
def quick_train():
    observer = Observer(); navigator = Navigator()
    file_emb = nn.Embedding(MAX_FILES, EMBED_DIM)
    fn_emb = nn.Embedding(MAX_FUNCTIONS, MAX_FUNCTIONS)
    ctx_emb = nn.Embedding(len(pb.CONTEXT_POOL), CONTEXT_EMBED)
    err_emb = nn.Embedding(MAX_ERRORS, ERROR_EMBED)
    bank = pb.TaskBank(size=2000)
    hist = pb.train(observer, navigator, bank, file_emb, fn_emb, ctx_emb, err_emb, epochs=3000)
    return observer, navigator, file_emb, fn_emb, ctx_emb, err_emb

print("Training Neuralese encoder/decoder...")
observer, navigator, file_emb, fn_emb, ctx_emb, err_emb = quick_train()

# --- Test tasks ---
TEST_TASKS = [
    ("src/module_3.py", 142, "replace", "evaluate", "TypeError: expected str, got int"),
    ("lib/utils_1.py", 89, "insert", "validate_input", "KeyError: config not found"),
    ("tests/test_4.py", 256, "delete", "process_batch", "IndexError: list index out of range"),
    ("src/module_7.py", 401, "append", "forward", "RuntimeError: tensor shape mismatch"),
    ("lib/utils_3.py", 15, "prepend", "load_config", "FileNotFoundError: config.yaml"),
]

def encode_task(file, line, edit, fn, err):
    """Encode a task into Neuralese vector."""
    bank = pb.TaskBank(size=1)  # dummy
    task = (file, line, edit, fn,
            [pb.CONTEXT_POOL[0]]*CONTEXT_LINES,
            [pb.CONTEXT_POOL[1]]*CONTEXT_LINES, err)
    x = bank.to_observer_input([task], file_emb, fn_emb, ctx_emb, err_emb)
    with torch.no_grad():
        z = observer(x)
    return z.squeeze(0).tolist()

def decode_task(z_vector):
    """Decode Neuralese vector back to task prediction."""
    z = torch.tensor([z_vector], dtype=torch.float32)
    with torch.no_grad():
        pf, pl, pe = navigator(z)
    file_idx = pf.argmax(-1).item()
    edit_idx = pe.argmax(-1).item()
    line_num = int(pl.item() * LINE_MAX)
    return FILES[file_idx], line_num, EDIT_TYPES[edit_idx]


# --- DeepSeek API helper ---
# NOTE: Requires DEEPSEEK_API_KEY in environment or set below.
# For security, never commit real API keys. Use env vars or a .env file.

API_KEY = None  # Set via environment: os.getenv("DEEPSEEK_API_KEY")

def call_deepseek(prompt, max_tokens=200):
    """Call DeepSeek API."""
    import urllib.request
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"], result["usage"]["total_tokens"]
    except Exception as e:
        return f"ERROR: {e}", 0


# --- Token Efficiency Benchmark ---

print("\n" + "=" * 60)
print("TOKEN EFFICIENCY BENCHMARK")
print("=" * 60)

for file, line, edit, fn, err in TEST_TASKS:
    # Text context (what Hermes currently sends to subagents)
    text_context = f"""File: {file}, line {line}
Edit: {edit} the code at this location
Function: {fn}
Error: {err}
Context lines show the surrounding code. Fix the bug."""

    # Neuralese encoding
    z = encode_task(file, line, edit, fn, err)
    decoded_file, decoded_line, decoded_edit = decode_task(z)

    # Token counts
    text_tokens = len(text_context.split()) * 1.3  # Approximate token count
    neuralese_chars = len(json.dumps(z))
    neuralese_tokens = neuralese_chars / 4  # ~4 chars per token

    file_match = "✓" if decoded_file == file else "✗"
    edit_match = "✓" if decoded_edit == edit else "✗"
    line_err = abs(decoded_line - line)

    print(f"\n  Task: {file}:{line} {edit} ({fn})")
    print(f"  Text tokens:   ~{text_tokens:.0f}")
    print(f"  Neuralese:     ~{neuralese_tokens:.0f} tokens ({neuralese_chars} chars)")
    print(f"  Savings:       {text_tokens/(neuralese_tokens+0.01):.0f}x")
    print(f"  File: {file_match}  Edit: {edit_match}  Line err: {line_err}")


# --- Cross-Model Integration Test ---

print("\n" + "=" * 60)
print("CROSS-MODEL TEST: DeepSeek → Neuralese → DeepSeek")
print("=" * 60)

# Task: DeepSeek plans a fix, we compress the context, DeepSeek executes
test_task = ("src/module_3.py", 142, "replace", "evaluate", "TypeError: expected str, got int")

# Mode 1: Full text context
text_ctx = f"""You are a code editor. Fix this bug:
FILE: {test_task[0]}
LINE: {test_task[1]}
EDIT: {test_task[2]} the line
FUNCTION: {test_task[3]}
ERROR: {test_task[4]}
Respond with exactly: FILE:<path> LINE:<num> ACTION:<edit>"""

# Mode 2: Neuralese context
z_vec = encode_task(*test_task)
df, dl, de = decode_task(z_vec)
neuralese_ctx = f"""You are a code editor. Fix this bug using the decoded Neuralese context:
FILE: {df}
LINE: {dl}
EDIT: {de}
ERROR: {test_task[4]}
Respond with exactly: FILE:<path> LINE:<num> ACTION:<edit>"""

print(f"\n  Original task: {test_task[0]}:{test_task[1]} {test_task[2]}")
print(f"  Decoded:       {df}:{dl} {de}")
print(f"  Line error:    {abs(dl - test_task[1])}")

print(f"\n  [Mode 1] Full text context:")
resp1, tokens1 = call_deepseek(text_ctx)
print(f"  Response: {resp1.strip()[:200]}")
print(f"  Tokens: {tokens1}")

print(f"\n  [Mode 2] Neuralese context:")
resp2, tokens2 = call_deepseek(neuralese_ctx)
print(f"  Response: {resp2.strip()[:200]}")
print(f"  Tokens: {tokens2}")

print(f"\n  Token comparison:")
print(f"  Mode 1 (text):    {tokens1} tokens")
print(f"  Mode 2 (Neuralese): {tokens2} tokens (context) + ~{len(json.dumps(z_vec))//4} tokens (vector)")
print(f"  Same action: {'YES' if resp1.strip()[:50] == resp2.strip()[:50] else 'CHECK ABOVE'}")

print("\nDone!")
