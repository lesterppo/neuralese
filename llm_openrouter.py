"""
Neuralese LLM Referential Game — OpenRouter Free Tier
======================================================
Tests whether a SINGLE LLM (Hermes 3 405B) can communicate through 16D.

The same model acts as both Sender and Receiver.
Tests: a) encode function → 16D vector, b) decode 16D → identify target.
"""
import json, time, urllib.request, sys, os, re, numpy as np
from pathlib import Path

# ---- Setup ----
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        if "OPENROUTER_API_KEY" in line and not line.startswith("#"):
            API_KEY=line.strip().split("=", 1)[1]; break

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"
LATENT_DIM = 16
N_CANDIDATES = 4

FUNCTIONS = [
    ("merge_dicts","Combines two dicts, d2 overrides d1 on key conflict. Returns new merged dict."),
    ("filter_positive","Keeps list elements > 0. Uses [x for x in items if x>0] comprehension."),
    ("flatten_nested","Flattens list-of-lists: [x for sub in nested for x in sub]."),
    ("group_by_key","Groups dicts by key: defaultdict(list), append pattern. Returns dict of lists."),
    ("normalize_data","(x-mean)/std for each element. Zero mean, unit variance output."),
    ("tokenize_text","Lowercases, removes commas, splits on whitespace. Returns word tokens."),
    ("camel_to_snake","Inserts _ before uppercase then lowercases. CamelCase → snake_case."),
    ("truncate_string","s[:n]+'...' if len(s)>n else s. Returns truncated or original."),
    ("binary_search","lo,hi=0,len-1; mid=(lo+hi)//2; compare, narrow, return index or -1."),
    ("top_k_items","sorted(items,key=fn,reverse=True)[:k]. Returns k highest scored."),
    ("deduplicate_list","seen=set(); yield if not in seen. Preserves order, removes dupes."),
    ("chunk_sequence","[seq[i:i+n] for i in range(0,len,n)]. Splits into fixed-size chunks."),
    ("read_json_file","with open(path) as f: return json.load(f). Parses JSON to Python object."),
    ("write_csv_rows","csv.writer(f).writerows(rows). Writes list of rows to CSV file."),
    ("safe_remove_file","os.path.exists(path) and os.remove(path). Returns bool success."),
    ("shuffle_list","r=lst[:]; random.shuffle(r); return r. Copies then shuffles."),
    ("pick_random_items","random.sample(lst, min(n,len(lst))). Picks without replacement."),
    ("frequency_count","Counter(items), return dict. Counts occurrences of each item."),
    ("merge_sorted_lists","Two-pointer merge: i,j=0,0; append smaller; handle remainders."),
    ("cache_lru_get","cache.move_to_end(key) if hit, return cache[key] else None. No insert."),
]


def call_llm(prompt, max_tokens=100, temperature=0.0):
    """Call OpenRouter with retry on rate limit."""
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/lesterppo/neuralese",
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(API_URL, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()
            tokens = result.get("usage", {}).get("total_tokens", 0)
            return content, tokens
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = (attempt + 1) * 8
                print(f"  (rate limit, waiting {wait}s...)", end="", flush=True)
                time.sleep(wait)
            elif attempt < 3:
                time.sleep(3)
    return f"ERROR: {err}", 0


def sender_prompt(func):
    name, desc = func
    return f"""Encode this Python function into exactly 16 floating-point numbers for AI-to-AI communication.

FUNCTION: {name}
WHAT IT DOES: {desc}

Use this encoding scheme:
- dims 0-3: category (+1=data, -1=string, +1=math, -1=io)
- dims 4-7: input type (+1=list, -1=dict, +1=file, -1=text)
- dims 8-11: output type (+1=list, -1=dict, +1=bool, -1=None)
- dims 12-15: complexity (-1=O(1), +1=O(n))

Output ONLY a JSON array of 16 floats. Example: [0.5, -0.3, 0.8, -0.2, 0.1, 0.9, -0.5, 0.0, 0.3, -0.7, 0.6, -0.1, 0.4, -0.8, 0.2, 0.9]

JSON:"""


def receiver_prompt(z_vector, idxs):
    vector_str = json.dumps([round(x, 4) for x in z_vector])
    cand_str = "\n".join([f"  {i+1}. {FUNCTIONS[idx][0]} — {FUNCTIONS[idx][1][:60]}"
                          for i, idx in enumerate(idxs)])
    return f"""You received a 16D vector encoding a function. Identify which candidate it matches.

VECTOR: {vector_str}

CANDIDATES:
{cand_str}

Compare the vector values to what each function would produce. Which candidate is it?

Reply with ONLY the number 1-4.

Answer:"""


def run_game(use_null=False):
    idxs = np.random.choice(len(FUNCTIONS), N_CANDIDATES, replace=False)
    target_idx = np.random.randint(0, N_CANDIDATES)
    target_func = FUNCTIONS[idxs[target_idx]]

    # Sender
    s_resp, s_tok = call_llm(sender_prompt(target_func), max_tokens=100)

    if use_null:
        z_vec = list(np.round(np.random.randn(LATENT_DIM) * 0.5, 4))
    else:
        try:
            m = re.search(r'\[[\d\s,.\-]+\]', s_resp)
            z_vec = json.loads(m.group()) if m else [0.0]*LATENT_DIM
            if len(z_vec) != LATENT_DIM: z_vec = [0.0]*LATENT_DIM
        except:
            z_vec = [0.0]*LATENT_DIM

    # Receiver
    r_resp, r_tok = call_llm(receiver_prompt(z_vec, idxs), max_tokens=10)

    # Parse
    try:
        nums = re.findall(r'\b([1-4])\b', r_resp)
        if nums:
            return int(nums[0])-1 == target_idx, s_tok + r_tok
    except: pass
    return False, s_tok + r_tok


# ---- MAIN ----
if __name__ == "__main__":
    print("=" * 70)
    print(f"NEURALESE LLM GAME — {MODEL}")
    print("=" * 70)
    print(f"  Candidates: {N_CANDIDATES} (chance=25%), Bottleneck: {LATENT_DIM}D")
    print()

    N = 6  # games
    correct = 0; null_correct = 0; tokens = 0

    print(f"[Running {N} games + {N} null-channel...]")
    for i in range(N):
        print(f"  Game {i+1}/{N}...", end=" ", flush=True)
        ok, tok = run_game(use_null=False)
        nok, ntok = run_game(use_null=True)
        if ok: correct += 1
        if nok: null_correct += 1
        tokens += tok + ntok
        print(f"signal={'✓' if ok else '✗'} null={'✓' if nok else '✗'} tok={tok+ntok}", flush=True)

    acc = correct / N
    null_acc = null_correct / N
    print(f"\n{'='*50}")
    print(f"RESULTS ({MODEL.split('/')[-1][:20]})")
    print(f"{'='*50}")
    print(f"  Accuracy:      {acc:.0%} ({correct}/{N})")
    print(f"  Null channel:  {null_acc:.0%} ({null_correct}/{N})")
    print(f"  Chance:        25%")
    print(f"  Over chance:   {acc-0.25:+.0%}")
    print(f"  Avg tokens:    {tokens/(N*2):.0f}")
    print(f"  Verdict:       {'EMERGENT' if acc > 0.40 and acc - null_acc > 0.1 else 'INCONCLUSIVE' if acc > 0.30 else 'NO SIGNAL'}")

    # Save
    with open("output/llm_results.json", "w") as f:
        json.dump({"model": MODEL, "games": N, "accuracy": acc,
                   "null_accuracy": null_acc, "tokens": tokens}, f, indent=2)
    print(f"\nSaved: output/llm_results.json")
