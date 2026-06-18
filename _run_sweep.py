import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from referential_game import FunctionBank
from exploration import make_sender, make_receiver, train_one, evaluate_one
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

out_dir = Path('output'); out_dir.mkdir(exist_ok=True)
dims = [2, 4, 6, 8, 12, 16, 24, 32]
results = []

with open(out_dir / 'sweep_log.txt', 'w') as log:
    for ld in dims:
        log.write(f'{ld}D... '); log.flush()
        train_bank = FunctionBank(500)
        test_bank = FunctionBank(200)
        s = make_sender(ld)
        r = make_receiver(ld, 4)
        best = train_one(s, r, train_bank, ld, 4, epochs=3000)
        ev = evaluate_one(s, r, test_bank, ld, 4, n_games=200)
        bits = ev['accuracy'] * 2.0
        r_ = {'dim': ld, 'best': best, 'test': ev['accuracy'], 'null': ev['null_accuracy'],
              'over': ev['over_chance'], 'bits': bits, 'bpd': bits/ld if ld else 0}
        results.append(r_)
        log.write(f'best={best:.1%} test={ev["accuracy"]:.1%} null={ev["null_accuracy"]:.1%}\n'); log.flush()
    
    log.write('\nRESULTS:\n')
    for r in results:
        log.write(f'{r["dim"]:>3}D: test={r["test"]:.1%} null={r["null"]:.1%} over={r["over"]:+.1%} bits/dim={r["bpd"]:.3f}\n')

# Plot
fig, ax = plt.subplots(figsize=(10,5))
dx = [r['dim'] for r in results]
ax.plot(dx, [r['test'] for r in results], 'b-o', markersize=8, label='Test accuracy')
ax.plot(dx, [r['null'] for r in results], 'r--s', markersize=8, label='Null channel')
ax.axhline(y=0.25, color='gray', ls=':', alpha=0.5, label='Chance (25%)')
ax.axhline(y=0.50, color='green', ls='--', alpha=0.3, label='50%')
ax.set_xlabel('Bottleneck dimension'); ax.set_ylabel('Accuracy')
ax.set_title('Neuralese Bottleneck Sweep'); ax.legend(); ax.grid(True,alpha=0.3)
ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig(out_dir/'bottleneck_sweep.png',dpi=150); plt.close()
print(f'Saved plots', flush=True)
