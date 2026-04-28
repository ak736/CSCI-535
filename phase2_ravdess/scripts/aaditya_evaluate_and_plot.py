#!/usr/bin/env python3
"""
aaditya_evaluate_and_plot.py — Phase 2 Step 6 (Aaditya)

Compute final evaluation metrics per ablation config and render all plots.

Metrics per config (A, B, C, D):
    - ROC-AUC (primary)
    - Precision / Recall / F1 at optimal threshold (Youden's J)
    - Confusion matrix

Plots (into phase2_ravdess/results/plots/):
    - roc_curves.png           — overlaid ROC curves for configs A-E with AUC in legend
    - ablation_delta.png       — bar chart of AUC gain per added feature group
    - emotion_pair_heatmap.png — heatmap by (audio_emotion, video_emotion)
    - confusion_matrix.png     — best config binary confusion matrix
    - score_distribution.png   — JSD distributions for congruent vs incongruent pairs

INPUTS:
    phase2_ravdess/predictions/predictions.csv
    phase2_ravdess/predictions/experiment_log.csv
    phase2_ravdess/scores/pair_scores.csv

OUTPUTS:
    phase2_ravdess/results/evaluation_metrics.csv
    phase2_ravdess/results/plots/*.png

RUN:
    python3 phase2_ravdess/scripts/aaditya_evaluate_and_plot.py \\
        --predictions    phase2_ravdess/predictions/predictions.csv \\
        --experiment_log phase2_ravdess/predictions/experiment_log.csv \\
        --pair_scores    phase2_ravdess/scores/pair_scores.csv \\
        --out_metrics    phase2_ravdess/results/evaluation_metrics.csv \\
        --plots_dir      phase2_ravdess/results/plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

JS_COLOR   = "#2563EB"  # blue
BASE_COLOR = "#DC2626"  # red
DIAG_COLOR = "#9CA3AF"  # gray

RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
}

CONFIGS = ["A", "B", "C", "D", "E"]

METRIC_COLUMNS = [
    "config", "auc_mean", "auc_std",
    "f1_mean", "precision_mean", "recall_mean",
]

PREDICTION_COLUMNS = ["pair_id", "label", "pred_prob", "pred_label", "config", "fold_actor"]
PAIR_SCORE_COLUMNS = ["audio_emotion_4class", "video_emotion_4class", "label", "jsd_audio_video", "jsd_composite"]
EMOTION_ORDER = ["happy", "angry", "sad", "neutral"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Fail fast if a required CSV is missing columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {name} missing required columns {missing}", file=sys.stderr)
        sys.exit(1)


def sort_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows in canonical ablation order A, B, C, D."""
    out = df.copy()
    out["config"] = pd.Categorical(out["config"], categories=CONFIGS, ordered=True)
    out = out.sort_values("config").reset_index(drop=True)
    out["config"] = out["config"].astype(str)
    return out


def compute_metrics_per_config(experiment_log_df: pd.DataFrame) -> pd.DataFrame:
    """Filter experiment_log to CONFIGS rows and sort in canonical order."""
    require_columns(experiment_log_df, METRIC_COLUMNS, "experiment_log")
    # Keep only the primary configs (ignore C_bal, E_bal, C_no_happy, etc.)
    metrics_df = experiment_log_df[experiment_log_df["config"].isin(CONFIGS)][METRIC_COLUMNS].copy()
    metrics_df = sort_by_config(metrics_df)
    missing = set(CONFIGS) - set(metrics_df["config"])
    if missing:
        print(f"ERROR: experiment_log missing configs {missing}", file=sys.stderr)
        sys.exit(1)
    if metrics_df.isna().sum().sum() != 0:
        print("ERROR: experiment_log contains NaN values", file=sys.stderr)
        sys.exit(1)
    return metrics_df


def compute_confusion_matrix(preds_df: pd.DataFrame, config: str) -> np.ndarray:
    """Binary confusion matrix (2x2) for the given config at threshold 0.5."""
    from sklearn.metrics import confusion_matrix

    sub = preds_df[preds_df["config"] == config]
    if sub.empty:
        print(f"ERROR: no prediction rows found for config {config}", file=sys.stderr)
        sys.exit(1)
    return confusion_matrix(sub["label"], sub["pred_label"], labels=[0, 1])


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_roc_curves(preds_df: pd.DataFrame, out_path: str) -> None:
    """Overlay ROC curves for configs A-D with AUC in the legend."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    palette = [JS_COLOR, "#0F766E", "#7C3AED", BASE_COLOR, "#EA580C"]
    for cfg, color in zip(CONFIGS, palette):
        sub = preds_df[preds_df["config"] == cfg]
        fpr, tpr, _ = roc_curve(sub["label"], sub["pred_prob"])
        auc = roc_auc_score(sub["label"], sub["pred_prob"])
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"Config {cfg} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color=DIAG_COLOR, ls=":", lw=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Ablation Configurations")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation_delta(log_df: pd.DataFrame, out_path: str) -> None:
    """Bar chart showing AUC increase going from A → B → C → D."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    plot_df = sort_by_config(log_df[["config", "auc_mean"]])
    sns.barplot(data=plot_df, x="config", y="auc_mean", color=JS_COLOR, ax=ax)

    for idx, row in plot_df.iterrows():
        ax.text(idx, row["auc_mean"] + 0.006, f"{row['auc_mean']:.3f}",
                ha="center", va="bottom", fontsize=10)
        if idx > 0:
            delta = row["auc_mean"] - plot_df.loc[idx - 1, "auc_mean"]
            ax.text(idx - 0.5, max(row["auc_mean"], plot_df.loc[idx - 1, "auc_mean"]) + 0.02,
                    f"+{delta:.3f}", ha="center", va="bottom", fontsize=9, color="#374151")

    ax.set_ylim(0.0, max(0.7, float(plot_df["auc_mean"].max()) + 0.08))
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_title("Ablation Study — Mean ROC-AUC by Configuration")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_emotion_pair_heatmap(pair_scores_df: pd.DataFrame, out_path: str) -> None:
    """Heatmap of mean jsd_audio_video by (audio_emotion_4class, video_emotion_4class)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 5))

    heatmap_df = pair_scores_df.pivot_table(
        index="audio_emotion_4class",
        columns="video_emotion_4class",
        values="jsd_audio_video",
        aggfunc="mean",
    ).reindex(index=EMOTION_ORDER, columns=EMOTION_ORDER)

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Mean JSD"},
        ax=ax,
    )
    ax.set_xlabel("Video emotion")
    ax.set_ylabel("Audio emotion")
    ax.set_title("Mean Audio-Video JSD by Emotion Pair")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Binary 2x2 confusion matrix heatmap (congruent/incongruent)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Congruent", "Incongruent"],
        yticklabels=["Congruent", "Incongruent"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_distribution(pair_scores_df: pd.DataFrame, out_path: str) -> None:
    """Violin/box plot of jsd_composite split by label (0 congruent, 1 incongruent)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    plot_df = pair_scores_df[["label", "jsd_composite"]].copy()
    plot_df["pair_label"] = plot_df["label"].map({0: "Congruent", 1: "Incongruent"})

    sns.violinplot(
        data=plot_df,
        x="pair_label",
        y="jsd_composite",
        hue="pair_label",
        palette=[BASE_COLOR, JS_COLOR],
        inner=None,
        cut=0,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x="pair_label",
        y="jsd_composite",
        hue="pair_label",
        width=0.25,
        showfliers=False,
        dodge=False,
        legend=False,
        boxprops={"facecolor": "white", "alpha": 0.8},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "#111827", "linewidth": 2},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Composite JSD Score")
    ax.set_title("Composite JSD by Pair Label")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(predictions_path: str, experiment_log_path: str, pair_scores_path: str,
        out_metrics_path: str, plots_dir: str) -> None:
    """Compute metrics, save metrics CSV, render all plots."""
    for label, p in [("predictions", predictions_path),
                     ("experiment_log", experiment_log_path),
                     ("pair_scores", pair_scores_path)]:
        if not os.path.isfile(p):
            print(f"ERROR: {label} CSV not found: {p}", file=sys.stderr)
            sys.exit(1)

    preds_df       = pd.read_csv(predictions_path)
    experiment_log = pd.read_csv(experiment_log_path)
    pair_scores_df = pd.read_csv(pair_scores_path)

    require_columns(preds_df, PREDICTION_COLUMNS, "predictions")
    require_columns(experiment_log, METRIC_COLUMNS, "experiment_log")
    require_columns(pair_scores_df, PAIR_SCORE_COLUMNS, "pair_scores")

    print(f"  predictions    : {len(preds_df)} rows")
    print(f"  experiment_log : {len(experiment_log)} rows")
    print(f"  pair_scores    : {len(pair_scores_df)} rows")

    os.makedirs(os.path.dirname(os.path.abspath(out_metrics_path)), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metrics_df = compute_metrics_per_config(experiment_log)
    metrics_df.to_csv(out_metrics_path, index=False)
    print(f"WROTE {out_metrics_path} {len(metrics_df)} rows")

    best_config = metrics_df.loc[metrics_df["auc_mean"].idxmax(), "config"]
    print(f"  best config (by AUC): {best_config}")
    cm = compute_confusion_matrix(preds_df, best_config)

    # Plots
    plot_roc_curves(preds_df, os.path.join(plots_dir, "roc_curves.png"))
    plot_ablation_delta(metrics_df, os.path.join(plots_dir, "ablation_delta.png"))
    plot_emotion_pair_heatmap(pair_scores_df, os.path.join(plots_dir, "emotion_pair_heatmap.png"))
    plot_confusion_matrix(cm, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_score_distribution(pair_scores_df, os.path.join(plots_dir, "score_distribution.png"))

    validate(out_metrics_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(metrics_path: str) -> None:
    """Assert metrics CSV has 4 rows (configs A-D), correct columns, no NaN."""
    df = pd.read_csv(metrics_path)

    print(f"\nVALIDATE {metrics_path}")
    print(f"  rows: {len(df)}")

    assert len(df) == len(CONFIGS), f"expected {len(CONFIGS)} rows, got {len(df)}"
    assert list(df.columns) == METRIC_COLUMNS, f"column mismatch: {list(df.columns)}"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert set(df["config"].tolist()) == set(CONFIGS), f"config column != {set(CONFIGS)}"

    print(f"  OK — {len(CONFIGS)} configs, all metrics present, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics + render all Phase 2 plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions",    default="phase2_ravdess/predictions/predictions_improved.csv")
    parser.add_argument("--experiment_log", default="phase2_ravdess/predictions/experiment_log_improved.csv")
    parser.add_argument("--pair_scores",    default="phase2_ravdess/scores/pair_scores_calibrated.csv")
    parser.add_argument("--out_metrics",    default="phase2_ravdess/results/evaluation_metrics_improved.csv")
    parser.add_argument("--plots_dir",      default="phase2_ravdess/results/plots")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out_metrics)
        return

    print("START aaditya_evaluate_and_plot")
    run(args.predictions, args.experiment_log, args.pair_scores,
        args.out_metrics, args.plots_dir)
    print("DONE")


if __name__ == "__main__":
    main()
