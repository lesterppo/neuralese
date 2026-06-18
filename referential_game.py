"""
Neuralese Referential Game — The Litmus Test
============================================
Canonical emergent communication test from linguistics/AI literature
(Lazaridou et al. 2017, Havrylov & Titov 2017).

V2: Actor-Critic (value baseline), 16D bottleneck, more function diversity.

SETUP:
  Sender sees RAW FUNCTION TEXT → outputs 16D Neuralese vector
  Receiver sees N CANDIDATES (hash embeddings, shuffled) + 16D vector
  Receiver must IDENTIFY which candidate is the target
  Both agents share reward: +1 correct, -1 wrong
  Actor-Critic with value baseline for stable training

WHY THIS IS THE LITMUS TEST:
  - Communication is the ONLY path to success.
  - Sender sees TEXT, Receiver sees HASH EMBEDDINGS — must develop shared protocol.
  - If accuracy exceeds chance, EMERGENT COMMUNICATION is proven.
"""

import torch, torch.nn as nn, torch.optim as optim, torch.distributions as D
import numpy as np, hashlib, uuid
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- CONFIG ---
LATENT_DIM = 16
HIDDEN = 128
NUM_CANDIDATES = 4
EMBED_DIM = 64
TRAIN_FUNCS = 800
TEST_FUNCS = 200
EPOCHS = 8000
BATCH = 32
LR = 1e-3
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5


# --- FUNCTION GENERATOR (more diverse) ---

def gen_function():
    """Generate a distinct Python function."""
    templates = [
        ("merge_dicts", ["d1", "d2"], "return {**d1, **d2}"),
        ("filter_positive", ["items"], "return [x for x in items if x > 0]"),
        ("flatten", ["nested"], "return [x for sub in nested for x in sub]"),
        ("group_by", ["records", "key"], "d={}; [d.setdefault(r[key],[]).append(r) for r in records]; return d"),
        ("normalize", ["data"], "m=sum(data)/len(data); s=(sum((x-m)**2 for x in data)/len(data))**0.5; return [(x-m)/s for x in data]"),
        ("tokenize", ["text"], "return text.lower().replace(',','').split()"),
        ("camel_case", ["name"], "parts=name.split('_'); return parts[0]+''.join(p.title() for p in parts[1:])"),
        ("snake_case", ["name"], "return ''.join('_'+c.lower() if c.isupper() else c for c in name).lstrip('_')"),
        ("truncate", ["s", "n"], "return s[:n]+'...' if len(s)>n else s"),
        ("read_json", ["path"], "import json; with open(path) as f: return json.load(f)"),
        ("write_csv", ["rows", "path"], "import csv; with open(path,'w') as f: csv.writer(f).writerows(rows)"),
        ("list_ext", ["dir", "ext"], "import os; return [f for f in os.listdir(dir) if f.endswith(ext)]"),
        ("safe_remove", ["path"], "import os; os.remove(path) if os.path.exists(path) else None"),
        ("cache_get", ["cache", "key", "maxsize"], "if key in cache: cache.move_to_end(key); return cache[key]; return None"),
        ("binsearch", ["arr", "x"], "lo,hi=0,len(arr)-1; while lo<=hi: m=(lo+hi)//2; v=arr[m]; return m if v==x else (lo:=m+1) if v<x else (hi:=m-1); return -1"),
        ("merge_sorted", ["a", "b"], "i=j=0; r=[]; while i<len(a) and j<len(b): r.append(a[i] if a[i]<b[j] else b[j]); i+=(a[i]<b[j]); j+=(a[i]>=b[j]); return r+a[i:]+b[j:]"),
        ("topk", ["items", "k", "keyfn"], "return sorted(items, key=keyfn, reverse=True)[:k]"),
        ("dedupe", ["items"], "seen=set(); return [x for x in items if not (x in seen or seen.add(x))]"),
        ("chunk", ["seq", "n"], "return [seq[i:i+n] for i in range(0,len(seq),n)]"),
        ("freq_count", ["items"], "from collections import Counter; return dict(Counter(items))"),
        ("shuffle_list", ["lst"], "import random; r=lst[:]; random.shuffle(r); return r"),
        ("pick_random", ["lst", "n"], "import random; return random.sample(lst, min(n,len(lst)))"),
        ("zip_with", ["a", "b", "fn"], "return [fn(x,y) for x,y in zip(a,b)]"),
        ("first_match", ["items", "pred"], "for i,x in enumerate(items): return i if pred(x) else None"),
    ]
    name, params, body = templates[np.random.randint(0, len(templates))]
    variants = [
        f"def {name}({', '.join(params)}):\n    \"\"\"{name}\"\"\"\n    {body}",
        f"def {name}({', '.join(params)}):\n    #{name}\n    {body}",
        f"def {name}({', '.join(reversed(params))}):\n    {body}",
    ]
    return variants[np.random.randint(0, len(variants))]


def func_to_embedding(func_text):
    """Hash function text to fixed-dim embedding."""
    vec = torch.zeros(EMBED_DIM)
    for i in range(len(func_text) - 1):
        h = hash(func_text[i:i+2]) % EMBED_DIM
        vec[h] += 1.0
    norm = vec.norm() + 1e-8
    return vec / norm


class FunctionBank:
    def __init__(self, n):
        self.funcs = [gen_function() for _ in range(n)]
        self.embeddings = torch.stack([func_to_embedding(f) for f in self.funcs])

    def sample_game(self, batch_size):
        B = batch_size; N = NUM_CANDIDATES
        candidates = torch.zeros(B, N, EMBED_DIM)
        targets = torch.zeros(B, dtype=torch.long)
        target_texts = []
        for b in range(B):
            tidx = np.random.randint(0, len(self.funcs))
            target_texts.append(self.funcs[tidx])
            dist = []
            while len(dist) < N - 1:
                d = np.random.randint(0, len(self.funcs))
                if d != tidx: dist.append(d)
            all_idx = [tidx] + dist
            np.random.shuffle(all_idx)
            for n in range(N):
                candidates[b, n] = self.embeddings[all_idx[n]]
            targets[b] = all_idx.index(tidx)
        return candidates, targets, target_texts


# --- AGENTS ---

class Sender(nn.Module):
    """CNN over raw text → 16D Neuralese + value baseline."""
    def __init__(self, vocab_size=128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, 32)
        self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 7, padding=3)
        self.fc = nn.Sequential(nn.Linear(192, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.z_head = nn.Sequential(nn.Linear(HIDDEN, LATENT_DIM), nn.LayerNorm(LATENT_DIM))
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


class Receiver(nn.Module):
    """N candidates + 16D z → score per candidate."""
    def __init__(self):
        super().__init__()
        self.cand_net = nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(HIDDEN + LATENT_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, candidates, z):
        B, N, _ = candidates.shape
        h = self.cand_net(candidates.view(B*N, EMBED_DIM))
        z_exp = z.unsqueeze(1).expand(B, N, LATENT_DIM).reshape(B*N, LATENT_DIM)
        scores = self.scorer(torch.cat([h, z_exp], -1))
        return scores.view(B, N)


# --- TRAINING (Actor-Critic) ---

def train_game(sender, receiver, bank, epochs=EPOCHS):
    params = list(sender.parameters()) + list(receiver.parameters())
    opt = optim.Adam(params, lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"loss": [], "acc": [], "entropy": [], "value": []}
    best_acc = 0.0

    for ep in range(epochs):
        candidates, target_pos, target_texts = bank.sample_game(BATCH)

        z, values = sender(target_texts)
        logits = receiver(candidates, z)

        dist = D.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        rewards = (actions == target_pos).float() * 2.0 - 1.0  # +1 or -1
        entropy = dist.entropy().mean()

        # Actor-critic loss
        advantage = rewards - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = nn.MSELoss()(values, rewards)
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        acc = (actions == target_pos).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
        hist["entropy"].append(entropy.item())
        hist["value"].append(values.mean().item())
        if acc > best_acc: best_acc = acc

        if ep % 2000 == 0:
            print(f"  ep {ep:5d}: acc={acc:.1%} best={best_acc:.1%} loss={loss.item():.4f} ent={entropy.item():.3f} val={values.mean().item():.3f}")

    return hist


# --- EVALUATION ---

def evaluate_game(sender, receiver, bank, n_games=500):
    correct = 0; null_correct = 0
    for _ in range(n_games):
        candidates, target_pos, target_texts = bank.sample_game(1)
        with torch.no_grad():
            z, _ = sender(target_texts)
            logits = receiver(candidates, z)
            pred = logits.argmax(-1).item()
            z_null = torch.randn(1, LATENT_DIM) * 0.5
            pred_null = receiver(candidates, z_null).argmax(-1).item()
        if pred == target_pos[0].item(): correct += 1
        if pred_null == target_pos[0].item(): null_correct += 1
    return {"accuracy": correct/n_games, "null_accuracy": null_correct/n_games, "chance": 1/NUM_CANDIDATES}


def evaluate_generalization(sender, receiver, test_bank, n_games=500):
    correct = 0; null_correct = 0
    for _ in range(n_games):
        tidx = np.random.randint(0, len(test_bank.funcs))
        target_text = test_bank.funcs[tidx]
        dist = []
        while len(dist) < NUM_CANDIDATES - 1:
            d = np.random.randint(0, len(test_bank.funcs))
            if d != tidx: dist.append(d)
        all_idx = [tidx] + dist
        np.random.shuffle(all_idx)
        true_pos = all_idx.index(tidx)
        candidates = torch.zeros(1, NUM_CANDIDATES, EMBED_DIM)
        for n in range(NUM_CANDIDATES):
            candidates[0, n] = test_bank.embeddings[all_idx[n]]
        with torch.no_grad():
            z, _ = sender([target_text])
            pred = receiver(candidates, z).argmax(-1).item()
            pred_null = receiver(candidates, torch.randn(1, LATENT_DIM)*0.5).argmax(-1).item()
        if pred == true_pos: correct += 1
        if pred_null == true_pos: null_correct += 1
    return {"gen_accuracy": correct/n_games, "gen_null": null_correct/n_games, "chance": 1/NUM_CANDIDATES}


# --- MAIN ---

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"; out_dir.mkdir(exist_ok=True)
    print("=" * 70)
    print("NEURALESE REFERENTIAL GAME v2 — Actor-Critic")
    print("=" * 70)
    print(f"  Candidates: {NUM_CANDIDATES} (chance = {1/NUM_CANDIDATES:.0%})")
    print(f"  Bottleneck: {LATENT_DIM}D = {LATENT_DIM*32} bits")
    print(f"  Training: {EPOCHS} epochs, Actor-Critic with value baseline")
    print()

    train_bank = FunctionBank(TRAIN_FUNCS)
    test_bank = FunctionBank(TEST_FUNCS)
    print(f"[Setup] {len(train_bank.funcs)} train, {len(test_bank.funcs)} test functions")
    print(f"[Training]...")

    sender = Sender(); receiver = Receiver()
    hist = train_game(sender, receiver, train_bank)

    train_eval = evaluate_game(sender, receiver, train_bank)
    test_eval = evaluate_game(sender, receiver, test_bank)
    gen_eval = evaluate_generalization(sender, receiver, test_bank)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    for k, v in {"Train": train_eval, "Test": test_eval, "Generalize": gen_eval}.items():
        label = "accuracy" if k != "Generalize" else "gen_accuracy"
        print(f"  {k:15s}: {v[label]:.1%}  (null={v.get('null_accuracy', v.get('gen_null',0)):.1%})")
    print(f"  {'Chance':15s}: {train_eval['chance']:.1%}")

    verdict_acc = max(test_eval['accuracy'], gen_eval['gen_accuracy'])
    over_chance = verdict_acc - train_eval['chance']
    null_drop = verdict_acc - test_eval['null_accuracy']
    print(f"\n{'='*50}")
    print(f"VERDICT: ", end="")
    if verdict_acc > 0.50 and null_drop > 0.10:
        print("EMERGENT COMMUNICATION DETECTED")
    elif over_chance > 0.10:
        print("WEAK SIGNAL — protocol may be forming")
    else:
        print("NO EMERGENT COMMUNICATION")

    # Plot
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    axes[0,0].plot(hist["acc"], alpha=0.5, color='blue')
    axes[0,0].axhline(y=1/NUM_CANDIDATES, color='red', ls='--', alpha=0.5, label='Chance')
    axes[0,0].set_title("Accuracy"); axes[0,0].legend(); axes[0,0].grid(True,alpha=0.3)
    bars = axes[0,1].bar(["Train","Test","Gen.","Null","Chance"],
        [train_eval["accuracy"],test_eval["accuracy"],gen_eval["gen_accuracy"],
         test_eval["null_accuracy"],train_eval["chance"]],
        color=['blue','green','purple','red','gray'],alpha=0.7)
    for bar,v in zip(bars,[train_eval["accuracy"],test_eval["accuracy"],gen_eval["gen_accuracy"],test_eval["null_accuracy"],train_eval["chance"]]):
        axes[0,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{v:.1%}",ha='center',fontsize=8)
    axes[0,1].set_title("Breakdown"); axes[0,1].set_ylim(0,1.1); axes[0,1].grid(True,alpha=0.3,axis='y')
    axes[1,0].plot(hist["entropy"],alpha=0.5,color='orange',label='Entropy')
    axes[1,0].plot(hist["value"],alpha=0.5,color='green',label='Mean value')
    axes[1,0].set_title("Training"); axes[1,0].legend(); axes[1,0].grid(True,alpha=0.3)
    axes[1,1].text(0.1,0.5,
        f"REFERENTIAL GAME v2\n"
        f"{'='*25}\n"
        f"Candidates: {NUM_CANDIDATES}\n"
        f"Bottleneck: {LATENT_DIM}D\n"
        f"Train acc: {train_eval['accuracy']:.1%}\n"
        f"Test acc: {test_eval['accuracy']:.1%}\n"
        f"Gen. acc: {gen_eval['gen_accuracy']:.1%}\n"
        f"Null: {test_eval['null_accuracy']:.1%}\n"
        f"Chance: {train_eval['chance']:.1%}",
        fontfamily='monospace',fontsize=9,va='center',transform=axes[1,1].transAxes)
    axes[1,1].axis('off')
    plt.suptitle("Neuralese Referential Game — Emergent Communication Test",fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/"referential_game_results.png",dpi=150); plt.close()
    print(f"Saved: {out_dir}/referential_game_results.png")
