#!/usr/bin/env python3
"""
aaditya_evaluate_and_plot.py — Phase 2 Step 6 (Aaditya)

OWNER: Aaditya
STEP : 6 (Day 6-7) — WAITS for Aniket Step 5 (predictions.csv, experiment_log.csv)

Compute final evaluation metrics per ablation config and render all plots.

Metrics per config (A, B, C, D):
    - ROC-AUC (primary)
    - Precision / Recall / F1 at optimal threshold (Youden's J)
    - Confusion matrix

Plots (into phase2_ravdess/results/plots/):
    - roc_curves.png           — overlaid ROC curves for configs A-D with AUC in legend
    - ablation_delta.png       — bar chart of AUC gain per added feature group
    - emotion_pair_heatmap.png — heatmap by (audio_emotion, video_emotion)
    - confusion_matrix.png     — best config binary confusion matrix
    - score_distribution.png   — JSD distributions for congruent vs incongruent pairs

Matplotlib style MATCHES Phase 1 (phase1/scripts/plot_results.py):
    JS_COLOR   = "#2563EB"  (blue)
    BASE_COLOR = "#DC2626"  (red)
    monospace for formulas, publication quality, no icons.

INPUTS:
    phase2_ravdess/predictions/predictions.csv       (from Aniket Step 5)
    phase2_ravdess/predictions/experiment_log.csv    (from Aniket Step 5)
    phase2_ravdess/scores/pair_scores.csv            (for score distributions)

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
# Plot style (Phase 1 — do not change)
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

CONFIGS = ["A", "B", "C", "D"]

METRIC_COLUMNS = [
    "config", "auc_mean", "auc_std",
    "f1_mean", "precision_mean", "recall_mean",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics_per_config(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each config A-D, compute AUC (mean/std across folds) + F1/P/R at
    Youden's-J-optimal threshold.
    """
    # TODO(Aaditya): implement using sklearn:
    #   from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
    #   For each config:
    #     auc_per_fold = []
    #     for fold_actor in preds_df["fold_actor"].unique():
    #         subset = preds_df[(preds_df.config==cfg) & (preds_df.fold_actor==fold_actor)]
    #         if subset["label"].nunique() < 2: continue   # skip degenerate folds
    #         auc_per_fold.append(roc_auc_score(subset.label, subset.pred_prob))
    #     auc_mean = np.mean(auc_per_fold); auc_std = np.std(auc_per_fold)
    #
    #     Pool all folds for this config, compute Youden's J threshold on ROC,
    #     then P/R/F1 at that threshold.
    raise NotImplementedError("TODO: implement compute_metrics_per_config")


def compute_confusion_matrix(preds_df: pd.DataFrame, config: str) -> np.ndarray:
    """Binary confusion matrix (2x2) for the given config at threshold 0.5."""
    # TODO(Aaditya): from sklearn.metrics import confusion_matrix
    #   sub = preds_df[preds_df.config == config]
    #   return confusion_matrix(sub.label, sub.pred_label)
    raise NotImplementedError("TODO: implement compute_confusion_matrix")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_roc_curves(preds_df: pd.DataFrame, out_path: str) -> None:
    """Overlay ROC curves for configs A-D with AUC in the legend."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # TODO(Aaditya): for each config in CONFIGS:
    #   sub = preds_df[preds_df.config == cfg]
    #   fpr, tpr, _ = roc_curve(sub.label, sub.pred_prob)
    #   auc = roc_auc_score(sub.label, sub.pred_prob)
    #   ax.plot(fpr, tpr, lw=2, label=f"Config {cfg} (AUC={auc:.3f})")
    # ax.plot([0,1],[0,1], color=DIAG_COLOR, ls=":", lw=1.5)
    # ax.set_xlabel("False Positive Rate")
    # ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves — Ablation Configurations")
    # ax.legend(loc="lower right", fontsize=10)
    # fig.tight_layout()
    # fig.savefig(out_path)
    # plt.close(fig)

    print(f"TODO: implement plot_roc_curves → {out_path}")
    plt.close(fig)


def plot_ablation_delta(log_df: pd.DataFrame, out_path: str) -> None:
    """Bar chart showing AUC increase going from A → B → C → D."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # TODO(Aaditya): bar chart of log_df["auc_mean"] across configs,
    # annotate delta between successive configs.
    print(f"TODO: implement plot_ablation_delta → {out_path}")
    plt.close(fig)


def plot_emotion_pair_heatmap(pair_scores_df: pd.DataFrame, out_path: str) -> None:
    """Heatmap of mean jsd_audio_video by (audio_emotion_4class, video_emotion_4class)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 5))

    # TODO(Aaditya): pivot_table on pair_scores with
    #   index="audio_emotion_4class", columns="video_emotion_4class",
    #   values="jsd_audio_video", aggfunc="mean"
    # Then ax.imshow with colormap, annotate values.
    print(f"TODO: implement plot_emotion_pair_heatmap → {out_path}")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Binary 2x2 confusion matrix heatmap (congruent/incongruent)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # TODO(Aaditya): imshow(cm) with ticks ["congruent","incongruent"]
    print(f"TODO: implement plot_confusion_matrix → {out_path}")
    plt.close(fig)


def plot_score_distribution(pair_scores_df: pd.DataFrame, out_path: str) -> None:
    """Violin/box plot of jsd_composite split by label (0 congruent, 1 incongruent)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # TODO(Aaditya): ax.violinplot([scores_0, scores_1]) or box; label ticks.
    print(f"TODO: implement plot_score_distribution → {out_path}")
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

    print(f"  predictions    : {len(preds_df)} rows")
    print(f"  experiment_log : {len(experiment_log)} rows")
    print(f"  pair_scores    : {len(pair_scores_df)} rows")

    os.makedirs(os.path.dirname(os.path.abspath(out_metrics_path)), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # TODO(Aaditya): uncomment once metric helpers are implemented
    # metrics_df = compute_metrics_per_config(preds_df)
    # metrics_df = metrics_df[METRIC_COLUMNS]
    # metrics_df.to_csv(out_metrics_path, index=False)
    # print(f"WROTE {out_metrics_path} {len(metrics_df)} rows")
    #
    # best_config = metrics_df.loc[metrics_df["auc_mean"].idxmax(), "config"]
    # print(f"  best config (by AUC): {best_config}")
    # cm = compute_confusion_matrix(preds_df, best_config)

    print("TODO: implement — uncomment compute_metrics + plot calls")

    # Plots
    plot_roc_curves(preds_df, os.path.join(plots_dir, "roc_curves.png"))
    plot_ablation_delta(experiment_log, os.path.join(plots_dir, "ablation_delta.png"))
    plot_emotion_pair_heatmap(pair_scores_df, os.path.join(plots_dir, "emotion_pair_heatmap.png"))
    # plot_confusion_matrix(cm, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_score_distribution(pair_scores_df, os.path.join(plots_dir, "score_distribution.png"))

    # validate(out_metrics_path)


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
    assert set(df["config"].tolist()) == set(CONFIGS), "config column != {A,B,C,D}"

    print("  OK — 4 configs, all metrics present, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics + render all Phase 2 plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions",    default="phase2_ravdess/predictions/predictions.csv")
    parser.add_argument("--experiment_log", default="phase2_ravdess/predictions/experiment_log.csv")
    parser.add_argument("--pair_scores",    default="phase2_ravdess/scores/pair_scores.csv")
    parser.add_argument("--out_metrics",    default="phase2_ravdess/results/evaluation_metrics.csv")
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
