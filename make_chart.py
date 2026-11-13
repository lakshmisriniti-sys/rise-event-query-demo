"""Draws results/chart.png: test F1 per method and scenario, and training F1 per refinement round."""

import json

import matplotlib.pyplot as plt

with open("results/results.json", encoding="utf-8") as f:
    results = json.load(f)

names = list(results)
labels = {"S1_disk": "S1 disk", "S2_payment_retries": "S2 payment retries", "S3_slow_db": "S3 slow DB"}
methods = [("search", "Brute-force search", "#9aa5b1"), ("llm_first", "LLM, first try", "#a9c4f5"),
           ("llm_loop", "LLM + score feedback", "#5b8def"), ("llm_stats", "LLM + feedback + data tool", "#1f9d74"),
           ("hidden", "True pattern", "#2b2b2b")]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
w = 0.16
for i, (m, label, color) in enumerate(methods):
    axes[0].bar([x + (i - 2) * w for x in range(len(names))],
                [results[n]["test"][m]["f1"] for n in names], w, label=label, color=color,
                alpha=0.35 if m == "hidden" else 1)
axes[0].set_xticks(range(len(names)), [labels.get(n, n) for n in names])
axes[0].set_ylim(0, 1.05)
axes[0].set_title("F1 on held-out test traces")

for n, color in zip(names, ["#7a5195", "#e0863b", "#1f9d74"]):
    for key, style, suffix in [("history", "--", " (feedback only)"), ("history_stats", "-", " (+ data tool)")]:
        h = results[n][key]
        axes[1].plot([r["round"] for r in h], [r["f1"] for r in h], style, marker="o", color=color,
                     label=labels.get(n, n) + suffix, alpha=0.55 if style == "--" else 1)
axes[1].set_xlabel("Refinement round")
axes[1].set_ylabel("Training F1")
axes[1].set_ylim(0, 1.05)
axes[1].set_xticks(range(1, 6))
axes[1].set_title("LLM query quality per round")
axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))

handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig("results/chart.png", dpi=150)
print("Saved results/chart.png")
