"""Turn results_<GPU>.txt into the slide table + a bar chart PNG.

Usage: python make_chart.py [results file] [output png prefix]
"""
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results_NVIDIA_GeForce_RTX_4070.txt"
prefix = sys.argv[2] if len(sys.argv) > 2 else "gpu_chart"

rows = defaultdict(lambda: defaultdict(list))  # mode -> bs -> [kernel times]
checksums = set()
gpu_name = ""
for line in open(path, encoding="utf-8", errors="replace"):
    if line.startswith("# GPU:"):
        gpu_name = line.split(":", 1)[1].split(",")[0].strip()
        continue
    m = re.search(r"mode=(naive|tiled) BS=(\d+) kernel=([0-9.]+) .*checksum=(\d+)", line)
    if m:
        rows[m.group(1)][int(m.group(2))].append(float(m.group(3)))
        checksums.add(m.group(4))

bs_list = sorted({bs for mode in rows for bs in rows[mode]})
print(f"GPU: {gpu_name}")
print(f"distinct checksums: {sorted(checksums)}  (must be exactly one)")
print()

# Table
hdr = f"| Block size | Naive (s) | Tiled (s) | Speedup (naive/tiled) |"
print(hdr)
print("|---|---|---|---|")
avgs = {}
for bs in bs_list:
    n = sum(rows["naive"][bs]) / len(rows["naive"][bs])
    t = sum(rows["tiled"][bs]) / len(rows["tiled"][bs])
    avgs[bs] = (n, t)
    print(f"| {bs}x{bs} | {n:.6f} | {t:.6f} | {n / t:.2f}x |")

best_n = min(avgs, key=lambda b: avgs[b][0])
best_t = min(avgs, key=lambda b: avgs[b][1])
print(f"\nBest naive block size: {best_n}x{best_n} ({avgs[best_n][0]:.6f} s)")
print(f"Best tiled block size: {best_t}x{best_t} ({avgs[best_t][1]:.6f} s)")

# Chart
x = range(len(bs_list))
w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
naive_v = [avgs[b][0] * 1000 for b in bs_list]
tiled_v = [avgs[b][1] * 1000 for b in bs_list]
ax.bar([i - w / 2 for i in x], naive_v, w, label="Naive", color="#c0504d")
ax.bar([i + w / 2 for i in x], tiled_v, w, label="Tiled (shared memory)", color="#4f81bd")
for i, v in enumerate(naive_v):
    ax.text(i - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
for i, v in enumerate(tiled_v):
    ax.text(i + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{b}x{b}" for b in bs_list])
ax.set_ylabel("Kernel time (ms)")
ax.set_xlabel("Block / tile size")
ax.set_title(f"1024x1024 float matmul kernel time - {gpu_name}")
ax.legend()
fig.tight_layout()
out = f"{prefix}.png"
fig.savefig(out, dpi=150)
print(f"\nchart saved: {out}")
