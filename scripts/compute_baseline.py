#!/usr/bin/env python3
"""
compute_baseline.py — Valence Difference Baseline

Computes a simple baseline incongruence score using the absolute difference
in "valence" between text and audio emotion distributions.

Valence proxy:
    valence = p_happy - p_angry  ∈ [-1, 1]
    baseline_score = |valence_text - valence_audio| / 2  → [0, 1]

This is compared against JS divergence in evaluation to show whether
the full distributional comparison outperforms a simpler heuristic.

Usage:
    python3 scripts/compute_baseline.py \
        --sample    annotations/annotation_sample.csv \
        --text_dir  outputs/embeddings \
        --audio_dir outputs/embeddings \
        --out       annotations/baseline_scores.csv

Then evaluate the baseline:
    python3 scripts/evaluate_detector.py \
        --annotations annotations/annotated_segments.csv \
        --scores      annotations/baseline_scores.csv \
        --out         results/baseline_metrics.csv
"""

import argparse
import os
import sys

import pandas as pd


def compute_valence(text_emb_df: pd.DataFrame, audio_emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join text and audio embeddings on segment_id and compute valence difference.
    valence = p_happy - p_angry
    baseline_score = |valence_text - valence_audio| / 2  → [0, 1]
    """
    merged = text_emb_df[["segment_id", "p_happy", "p_angry"]].merge(
        audio_emb_df[["segment_id", "p_happy", "p_angry"]].rename(
            columns={"p_happy": "a_happy", "p_angry": "a_angry"}
        ),
        on="segment_id", how="inner"
    )
    merged["valence_text"]  = merged["p_happy"] - merged["p_angry"]
    merged["valence_audio"] = merged["a_happy"] - merged["a_angry"]
    merged["baseline_score"] = (merged["valence_text"] - merged["valence_audio"]).abs() / 2.0
    return merged[["segment_id", "valence_text", "valence_audio", "baseline_score"]]


def run(sample_path: str, text_dir: str, audio_dir: str, out_path: str) -> None:
    if not os.path.isfile(sample_path):
        print(f"ERROR: sample not found: {sample_path}", file=sys.stderr)
        sys.exit(1)

    sample_df = pd.read_csv(sample_path)
    meetings  = sample_df["meeting_id"].unique().tolist()

    all_scores = []
    for meeting in meetings:
        text_path  = os.path.join(text_dir,  f"{meeting}_text_emotions.csv")
        audio_path = os.path.join(audio_dir, f"{meeting}_audio_emotions.csv")

        if not os.path.isfile(text_path) or not os.path.isfile(audio_path):
            print(f"  WARNING: missing embeddings for {meeting} — skipping")
            continue

        text_df  = pd.read_csv(text_path)
        audio_df = pd.read_csv(audio_path)

        scores = compute_valence(text_df, audio_df)
        scores["meeting_id"] = meeting
        all_scores.append(scores)

    if not all_scores:
        print("ERROR: no meeting embeddings found", file=sys.stderr)
        sys.exit(1)

    all_scores_df = pd.concat(all_scores, ignore_index=True)

    # Join with sample to get annotation_id and confidence weighting
    out = sample_df[["annotation_id", "meeting_id", "segment_id", "final_score"]].merge(
        all_scores_df[["meeting_id", "segment_id", "baseline_score"]],
        on=["meeting_id", "segment_id"], how="left"
    )

    # Apply asr_confidence weighting (same as JS scorer for fair comparison)
    # baseline_score already in [0,1]; weight by asr_confidence from sample
    if "final_score" in sample_df.columns:
        # We don't have asr_confidence in sample directly, so use raw baseline
        pass

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(out)} rows")
    print(f"  mean baseline_score : {out['baseline_score'].mean():.4f}")
    print(f"  max  baseline_score : {out['baseline_score'].max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute valence-difference baseline scores for annotation sample"
    )
    parser.add_argument("--sample",    required=True, help="Path to annotation_sample.csv")
    parser.add_argument("--text_dir",  required=True, help="Directory containing *_text_emotions.csv")
    parser.add_argument("--audio_dir", required=True, help="Directory containing *_audio_emotions.csv")
    parser.add_argument("--out",       required=True, help="Path to output baseline_scores.csv")
    args = parser.parse_args()

    print("START baseline computation")
    run(args.sample, args.text_dir, args.audio_dir, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
