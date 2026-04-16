"""
Plot evaluation results for Text vs Audio Incongruence Scorer (Phase 1).
Outputs 5 slide-ready figures to results/plots/.

Run: python3 scripts/plot_results.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── output dir ────────────────────────────────────────────────────────────────
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})

JS_COLOR = "#2563EB"   # blue  — JS divergence (ours)
BASE_COLOR = "#DC2626"   # red   — valence baseline
DIAG_COLOR = "#9CA3AF"   # gray  — chance line

# ── load data ─────────────────────────────────────────────────────────────────
roc_js = pd.read_csv("results/evaluation_metrics_roc_curve.csv")
roc_base = pd.read_csv("results/baseline_metrics_roc_curve.csv")

metrics_js = pd.read_csv("results/evaluation_metrics.csv").iloc[0]
metrics_base = pd.read_csv("results/baseline_metrics.csv").iloc[0]

annotations = pd.read_csv("annotations/annotated_segments.csv")
scores_df = pd.read_csv("annotations/annotation_sample.csv")

# merge scores with ground-truth labels
merged = scores_df.merge(
    annotations[["annotation_id", "majority_label"]],
    on="annotation_id",
    how="inner"
)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curve Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5.5))

ax.plot(roc_js["fpr"],   roc_js["tpr"],
        color=JS_COLOR,   lw=2.5,
        label=f"JS Divergence (AUC = {metrics_js['roc_auc']:.3f})")
ax.plot(roc_base["fpr"], roc_base["tpr"],
        color=BASE_COLOR, lw=2.5, linestyle="--",
        label=f"Valence Difference (AUC = {metrics_base['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=DIAG_COLOR, lw=1.5, linestyle=":")

# mark the operating threshold (0.3) for JS
op = roc_js[roc_js["threshold"] <= 0.3 + 1e-9].iloc[0]
ax.scatter(op["fpr"], op["tpr"], s=90, zorder=5, color=JS_COLOR,
           edgecolors="white", linewidths=1.5)
ax.annotate("  threshold = 0.3", (op["fpr"], op["tpr"]),
            fontsize=9, color=JS_COLOR, va="center")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — JS Divergence vs Baseline")
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.fill_between(roc_js["fpr"], roc_js["tpr"], alpha=0.06, color=JS_COLOR)

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/fig1_roc_curve.png")
plt.close(fig)
print("Saved fig1_roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Metrics Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
metric_names = ["ROC-AUC", "Precision", "Recall", "F1-Score"]
js_vals = [metrics_js["roc_auc"],   metrics_js["precision"],
           metrics_js["recall"],    metrics_js["f1"]]
base_vals = [metrics_base["roc_auc"], metrics_base["precision"],
             metrics_base["recall"],  metrics_base["f1"]]

x = np.arange(len(metric_names))
w = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - w/2, js_vals,   w, color=JS_COLOR,
               label="JS Divergence (ours)", alpha=0.92)
bars2 = ax.bar(x + w/2, base_vals, w, color=BASE_COLOR,
               label="Valence Difference (baseline)", alpha=0.92)

# value labels on bars
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9)

# target line for AUC
ax.axhline(0.75, color="#6B7280", lw=1.2, linestyle="--", alpha=0.6)
ax.text(3.6, 0.755, "AUC target (0.75)", fontsize=8, color="#6B7280")

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics — JS Divergence vs Baseline")
ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/fig2_metrics_comparison.png")
plt.close(fig)
print("Saved fig2_metrics_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Incongruence Score Distribution by Ground-Truth Label
# ══════════════════════════════════════════════════════════════════════════════
aligned = merged[merged["majority_label"] == 0]["final_score"]
incongruent = merged[merged["majority_label"] == 1]["final_score"]

fig, ax = plt.subplots(figsize=(6, 5))

data_to_plot = [aligned.values, incongruent.values]
labels = [f"Aligned\n(n={len(aligned)})",
          f"Incongruent\n(n={len(incongruent)})"]
colors = ["#4ADE80", "#F97316"]

bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.45,
                medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)
for element in ["whiskers", "caps", "fliers"]:
    for item in bp[element]:
        item.set_color("#374151")

# scatter individual points (jittered)
for i, (data, color) in enumerate(zip(data_to_plot, colors), start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(data))
    ax.scatter(np.full(len(data), i) + jitter, data,
               alpha=0.35, s=18, color=color, zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels(labels)
ax.set_ylabel("Final Incongruence Score")
ax.set_title("Score Distribution by Ground-Truth Label")

# decision threshold line
ax.axhline(0.3, color=JS_COLOR, lw=1.5, linestyle="--", alpha=0.7)
ax.text(2.35, 0.305, "threshold = 0.3", fontsize=9, color=JS_COLOR)

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/fig3_score_distribution.png")
plt.close(fig)
print("Saved fig3_score_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Annotation Label Distribution (human ground truth)
# ══════════════════════════════════════════════════════════════════════════════
label_counts = annotations["majority_label"].value_counts().sort_index()
n_aligned = int(label_counts.get(0, 0))
n_incong = int(label_counts.get(1, 0))
total = n_aligned + n_incong

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

# — Pie chart
wedges, texts, autotexts = axes[0].pie(
    [n_aligned, n_incong],
    labels=["Aligned", "Incongruent"],
    colors=["#4ADE80", "#F97316"],
    autopct="%1.1f%%",
    startangle=130,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11)
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight("bold")
axes[0].set_title(f"Ground-Truth Label Split\n(n = {total} segments)")

# — Per-score-tier bar (low / medium / high)
# divide 180 annotations into terciles by annotation_id order (60 each)
ann_with_scores = annotations.merge(
    scores_df[["annotation_id", "final_score"]], on="annotation_id", how="inner"
)
ann_with_scores["tier"] = pd.cut(
    ann_with_scores["final_score"],
    bins=[ann_with_scores["final_score"].min() - 1e-9,
          ann_with_scores["final_score"].quantile(1/3),
          ann_with_scores["final_score"].quantile(2/3),
          ann_with_scores["final_score"].max() + 1e-9],
    labels=["Low\nScore", "Mid\nScore", "High\nScore"]
)

tier_counts = ann_with_scores.groupby(
    "tier")["majority_label"].value_counts().unstack(fill_value=0)
tier_counts.columns = ["Aligned", "Incongruent"]

x = np.arange(len(tier_counts))
w = 0.32
axes[1].bar(x - w/2, tier_counts["Aligned"],    w,
            color="#4ADE80", label="Aligned",    alpha=0.9)
axes[1].bar(x + w/2, tier_counts["Incongruent"], w,
            color="#F97316", label="Incongruent", alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(tier_counts.index)
axes[1].set_ylabel("Count")
axes[1].set_title("Incongruent Segments by Score Tier")
axes[1].legend(fontsize=10)
for i, row in tier_counts.iterrows():
    xi = list(tier_counts.index).index(i)
    axes[1].text(xi - w/2, row["Aligned"] + 0.3,
                 str(row["Aligned"]),    ha="center", fontsize=9)
    axes[1].text(xi + w/2, row["Incongruent"] + 0.3,
                 str(row["Incongruent"]), ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/fig4_annotation_distribution.png")
plt.close(fig)
print("Saved fig4_annotation_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-Meeting Incongruence Statistics
# ══════════════════════════════════════════════════════════════════════════════
meeting_stats = {
    "ES2002a": (496,  0.1678, 0.7715),
    "ES2002b": (1151, 0.1723, 0.7058),
    "ES2002c": (1281, 0.1319, 0.6973),
    "ES2002d": (1178, 0.1454, 0.7764),
    "ES2003a": (435,  0.1461, 0.6503),
    "ES2003b": (1189, 0.1230, 0.6723),
    "ES2003c": (1269, 0.1505, 0.7507),
    "ES2003d": (1208, 0.1635, 0.7640),
    "ES2004a": (460,  0.2321, 0.7467),
    "ES2004b": (1216, 0.1926, 0.7797),
    "ES2004c": (1203, 0.1846, 0.7429),
    "ES2004d": (990,  0.1511, 0.7425),
}

meetings = list(meeting_stats.keys())
mean_vals = [v[1] for v in meeting_stats.values()]
max_vals = [v[2] for v in meeting_stats.values()]
n_segs = [v[0] for v in meeting_stats.values()]

x = np.arange(len(meetings))
w = 0.38
fig, ax1 = plt.subplots(figsize=(12, 5))

bars_mean = ax1.bar(x - w/2, mean_vals, w, color=JS_COLOR,
                    label="Mean Score",  alpha=0.88)
bars_max = ax1.bar(x + w/2, max_vals,  w, color="#7C3AED",
                   label="Max Score",   alpha=0.88)

# overall mean reference line
overall_mean = 0.1600
ax1.axhline(overall_mean, color="#9CA3AF", lw=1.5, linestyle="--")
ax1.text(11.6, overall_mean + 0.005, f"overall\nmean {overall_mean:.3f}",
         fontsize=8, color="#6B7280", ha="right")

# segment count as secondary axis
ax2 = ax1.twinx()
ax2.plot(x, n_segs, "o--", color="#D97706", lw=1.8,
         ms=7, label="# Segments", alpha=0.85)
ax2.set_ylabel("Number of Segments", color="#D97706", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#D97706")
ax2.spines["top"].set_visible(False)

ax1.set_xticks(x)
ax1.set_xticklabels(meetings, rotation=35, ha="right")
ax1.set_ylabel("Incongruence Score")
ax1.set_title("Per-Meeting Incongruence Statistics (12 Meetings)")

# combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, framealpha=0.9)

plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/fig5_per_meeting_stats.png")
plt.close(fig)
print("Saved fig5_per_meeting_stats.png")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to: {PLOTS_DIR}/")
print("  fig1_roc_curve.png           — ROC curve (JS vs baseline)")
print("  fig2_metrics_comparison.png  — AUC / Precision / Recall / F1 bar chart")
print("  fig3_score_distribution.png  — Score box plots by ground-truth label")
print("  fig4_annotation_distribution.png — Label split + by-tier breakdown")
print("  fig5_per_meeting_stats.png   — Per-meeting mean/max scores + segment counts")
