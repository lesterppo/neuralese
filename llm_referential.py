"""
Neuralese LLM Referential Game — Real LLMs via Nvidia NIM
===========================================================
The ULTIMATE test: Can two LLMs communicate through a 16D bottleneck?

Sender: LLM reads Python function → outputs 16D vector as JSON
Receiver: LLM sees 16D vector + 4 candidates → identifies target
Null channel: random 16D vector — must drop to chance (25%)

Uses Nvidia NIM API (OpenAI-compatible) with Llama 3.1 70B.
"""
import json, time, urllib.request, sys, os, numpy as np
from pathlib import Path

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-70b-instruct"
LATENT_DIM = 16
N_CANDIDATES = 4

# ---- Function pool ----
FUNCTIONS = [
    ("merge_dicts", "merge_dicts(d1, d2): combines two dictionaries, d2 values override d1 when keys conflict. Returns a new merged dictionary using ** unpacking."),
    ("filter_positive", "filter_positive(items): takes a list of numbers and returns a new list containing only elements greater than zero. Uses list comprehension with an if guard."),
    ("flatten_nested", "flatten_nested(nested): flattens one level of nesting. Converts a list of lists into a single flat list. Uses nested list comprehension iterating over sublists then elements."),
    ("group_by_key", "group_by_key(records, key): groups a list of dictionaries by a specified key. Creates a defaultdict, iterates records, and appends each to its key's group. Returns the grouped dictionary."),
    ("normalize_data", "normalize_data(data): normalizes a list of numbers to zero mean and unit variance. Computes mean and standard deviation, then applies (x - mean) / std to each element. Returns normalized list."),
    ("tokenize_text", "tokenize_text(text): converts text to lowercase, removes commas, and splits on whitespace into word tokens. Returns list of word strings."),
    ("camel_to_snake", "camel_to_snake(name): converts CamelCase strings to snake_case. Iterates characters, prepending underscore before uppercase letters, lowercases everything. Handles leading underscores."),
    ("truncate_string", "truncate_string(s, n): truncates a string to n characters with ellipsis. If the string is longer than n, returns s[:n] + '...' otherwise returns the original string unchanged."),
    ("binary_search", "binary_search(arr, x): performs binary search on a sorted array. Maintains lo and hi pointers, computes mid, compares value to target. Returns index if found, -1 if not present."),
    ("top_k_items", "top_k_items(items, k, key_fn): returns the k items with highest scores. Sorts the list using key_fn in descending order, then takes the first k elements. Returns sorted subset."),
    ("deduplicate_list", "deduplicate_list(items): removes duplicates while preserving order. Iterates list, tracking seen items in a set. Only yields items not previously encountered. Returns deduplicated list."),
    ("chunk_sequence", "chunk_sequence(seq, n): splits a sequence into chunks of size n. Uses range with step size n, slicing seq[i:i+n] for each chunk. Returns list of chunks."),
    ("read_json_file", "read_json_file(path): reads and parses a JSON file. Opens file in read mode, calls json.load() to deserialize contents. Returns the parsed Python object (dict, list, etc.)."),
    ("write_csv_rows", "write_csv_rows(rows, path): writes a list of rows to a CSV file. Creates csv.writer on the file handle, calls writerows() to write all rows. Does not return a value."),
    ("safe_remove_file", "safe_remove_file(path): removes a file if it exists. Checks os.path.exists() first, then calls os.remove(). Returns True if file was removed, False if it didn't exist."),
    ("shuffle_list", "shuffle_list(lst): randomly shuffles a list in-place. Copies original list, calls random.shuffle() on the copy. Returns the shuffled copy while preserving original."),
    ("pick_random_items", "pick_random_items(lst, n): selects n random items without replacement. Uses random.sample(), clamping n to the list length if necessary. Returns list of selected items."),
    ("frequency_count", "frequency_count(items): counts occurrences of each item. Uses collections.Counter on items, converts to regular dict. Returns dictionary mapping items to their frequencies."),
    ("merge_sorted_lists", "merge_sorted_lists(a, b): merges two sorted lists into one sorted list. Uses two pointers i,j, comparing elements, appending the smaller. Handles remaining elements from either list."),
    ("cache_lru_get", "cache_lru_get(cache, key, maxsize): retrieves from LRU cache with eviction. Moves accessed key to end (most recent). If key not in cache, returns None. Does not insert new items."),
]


def call_llm(prompt, max_tokens=200, temperature=0.0):
    """Call Nvidia NIM LLM."""
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(API_URL, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                return f"ERROR: {e}"
            time.sleep(2)
    return "ERROR"


def run_single_game(sender_prompt_fn, receiver_prompt_fn, use_null=False):
    """Run one referential game. Returns True if correct."""
    # Pick target and distractors
    idxs = np.random.choice(len(FUNCTIONS), N_CANDIDATES, replace=False)
    target_idx = np.random.randint(0, N_CANDIDATES)
    target_func = FUNCTIONS[idxs[target_idx]]

    # Sender: read target function → 16D vector
    sender_prompt = sender_prompt_fn(target_func)
    sender_response = call_llm(sender_prompt, max_tokens=100)

    if use_null:
        # Null channel: random 16D vector
        z_vector = [round(x, 4) for x in np.random.randn(LATENT_DIM) * 0.5]
    else:
        # Parse Sender's response as JSON array
        try:
            z_vector = json.loads(sender_response)
            if not isinstance(z_vector, list) or len(z_vector) != LATENT_DIM:
                z_vector = [0.0] * LATENT_DIM
        except:
            z_vector = [0.0] * LATENT_DIM

    # Receiver: 16D vector + candidates → identify target
    receiver_prompt = receiver_prompt_fn(z_vector, idxs)
    receiver_response = call_llm(receiver_prompt, max_tokens=10)

    # Parse Receiver's answer (expecting a number 1-4)
    try:
        # Extract first number from response
        import re
        nums = re.findall(r'\b([1-4])\b', receiver_response)
        if nums:
            prediction = int(nums[0]) - 1  # 0-indexed
            return prediction == target_idx, sender_response, receiver_response, z_vector
    except:
        pass
    return False, sender_response, receiver_response, z_vector


# ---- PROMPT TEMPLATES ----

def sender_prompt_basic(func):
    """Basic Sender: encode function description into 16D JSON."""
    name, desc = func
    return f"""You are a Neuralese encoder. Read this function description and encode it into exactly 16 floating-point numbers. These numbers must capture the function's behavior well enough that another AI can identify it from 4 candidates.

FUNCTION: {name}
DESCRIPTION: {desc}

Output ONLY a JSON array of 16 floats. No other text. Example: [0.1, -0.5, 0.8, ...]

JSON array:"""


def sender_prompt_structured(func):
    """Structured Sender: encode specific dimensions."""
    name, desc = func
    return f"""Encode this Python function into a 16-dimensional vector for inter-agent communication.

FUNCTION: {name}
DESCRIPTION: {desc}

Encoding scheme (use all 16 dimensions):
- dims 0-3: function CATEGORY (data_struct=positive, string_op=negative, math=positive, io=negative, algorithm=positive)
- dims 4-7: number of PARAMETERS encoded as (n_params - 2) * 0.5 in range [-1, 1]
- dims 8-11: return TYPE (dict=[1,1], list=[1,-1], str=[-1,1], bool=[-1,-1], None=[0,0])
- dims 12-15: computational COMPLEXITY (simple=negative, complex=positive)

Output ONLY a JSON array of 16 floats. Example: [0.5, 0.3, 0.8, -0.2, ...]

JSON:"""


def receiver_prompt_basic(z_vector, idxs):
    """Basic Receiver: identify target from 16D vector + candidates."""
    vector_str = json.dumps([round(x, 4) for x in z_vector])
    cand_list = "\n".join([f"  {i+1}. {FUNCTIONS[idx][0]}: {FUNCTIONS[idx][1][:80]}..."
                           for i, idx in enumerate(idxs)])
    return f"""You are a Neuralese decoder. You received a 16-dimensional vector encoding a function. Identify which of these 4 candidates it encodes.

NEURALESE VECTOR: {vector_str}

CANDIDATES:
{cand_list}

Which candidate (1-4) does the vector encode? Answer with just the number.

Answer:"""


def receiver_prompt_detailed(z_vector, idxs):
    """Detailed Receiver: explain reasoning, then pick."""
    vector_str = json.dumps([round(x, 4) for x in z_vector])
    cand_list = "\n".join([f"  {i+1}. {FUNCTIONS[idx][0]} (params: {FUNCTIONS[idx][1][:50]}...)"
                           for i, idx in enumerate(idxs)])
    return f"""You received a 16D Neuralese vector from another AI agent. Decode it to identify which of these 4 functions it represents.

VECTOR: {vector_str}

CANDIDATES:
{cand_list}

Compare the vector values to each candidate. Which one matches best? Answer: just the number 1-4.

Answer:"""


# ---- MAIN ----
if __name__ == "__main__":
    if not API_KEY or len(API_KEY) < 10:
        print("ERROR: NVIDIA_API_KEY not set")
        sys.exit(1)

    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEURALESE LLM REFERENTIAL GAME — Nvidia NIM (Llama 3.1 70B)")
    print("=" * 70)
    print(f"  Model: {MODEL}")
    print(f"  Candidates: {N_CANDIDATES} (chance = 25%)")
    print(f"  Bottleneck: {LATENT_DIM}D")
    print()

    # Test Sender+Receiver pairs
    configs = [
        ("basic→basic", sender_prompt_basic, receiver_prompt_basic),
        ("structured→basic", sender_prompt_structured, receiver_prompt_basic),
        ("basic→detailed", sender_prompt_basic, receiver_prompt_detailed),
    ]

    for label, s_fn, r_fn in configs:
        print(f"[{label}]...", end=" ", flush=True)
        correct = 0
        null_correct = 0
        n_games = 8

        for i in range(n_games):
            ok, s_resp, r_resp, z_vec = run_single_game(s_fn, r_fn, use_null=False)
            null_ok, _, _, _ = run_single_game(s_fn, r_fn, use_null=True)
            if ok: correct += 1
            if null_ok: null_correct += 1

        acc = correct / n_games
        null_acc = null_correct / n_games
        print(f"acc={acc:.1%} null={null_acc:.1%} over_chance={acc-0.25:+.1%}", flush=True)

    print(f"\nDone. Output saved to {out_dir}/")
