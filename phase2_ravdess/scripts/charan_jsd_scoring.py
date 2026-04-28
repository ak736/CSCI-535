#!/usr/bin/env python3
"""
charan_jsd_scoring.py — Phase 2 Step 4 (Charan)

Compute Jensen–Shannon divergence between every pair of modalities
(text, audio, video) for every constructed pair.

JSD normalized to [0, 1] by dividing by log(2).

Four score columns per row:
    jsd_text_audio  = JS(P_text,  P_audio)
    jsd_text_video  = JS(P_text,  P_video)
    jsd_audio_video = JS(P_audio, P_video)
    jsd_composite   = mean of the three

INPUT:
    phase2_ravdess/pairs/incongruence_pairs.csv

OUTPUT:
    phase2_ravdess/scores/pair_scores.csv
    (= input CSV + 4 extra JSD columns appended)

RUN:
    python3 phase2_ravdess/scripts/charan_jsd_scoring.py \\
        --pairs phase2_ravdess/pairs/incongruence_pairs.csv \\
        --out   phase2_ravdess/scores/pair_scores.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_NAMES = ["happy", "angry", "sad", "neutral"]
AUDIO_COLS = [f"p_audio_{e}" for e in EMOTION_NAMES]
VIDEO_COLS = [f"p_video_{e}" for e in EMOTION_NAMES]
TEXT_COLS  = [f"p_text_{e}"  for e in EMOTION_NAMES]

JSD_COLUMNS = ["jsd_text_audio", "jsd_text_video", "jsd_audio_video", "jsd_composite"]


# ---------------------------------------------------------------------------
# JSD math
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL(P || Q) using natural log.
    eps added to avoid log(0) — both distributions should already be valid
    after normalization but guard anyway.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon divergence, normalized to [0, 1].

        M = 0.5 * (P + Q)
        JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    Raw range: [0, log(2)]
    Normalized: divide by log(2) → [0, 1]
    """
    m = 0.5 * (p + q)
    js_raw = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(js_raw / np.log(2))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    """Guard against float drift — ensure vec sums to 1.0, else uniform."""
    total = vec.sum()
    if total <= 0:
        return np.ones(4) / 4
    return vec / total


def compute_jsd_row(row: pd.Series) -> dict:
    """Compute the 4 JSD scores for a single pair row."""
    p_text  = _safe_normalize(np.array([row[c] for c in TEXT_COLS],  dtype=np.float64))
    p_audio = _safe_normalize(np.array([row[c] for c in AUDIO_COLS], dtype=np.float64))
    p_video = _safe_normalize(np.array([row[c] for c in VIDEO_COLS], dtype=np.float64))

    ta = js_divergence(p_text,  p_audio)
    tv = js_divergence(p_text,  p_video)
    av = js_divergence(p_audio, p_video)
    composite = (ta + tv + av) / 3.0

    return {
        "jsd_text_audio":  round(ta, 6),
        "jsd_text_video":  round(tv, 6),
        "jsd_audio_video": round(av, 6),
        "jsd_composite":   round(composite, 6),
    }


def run(pairs_path: str, out_path: str) -> None:
    """Load pairs CSV, compute 4 JSD scores per row, write pair_scores CSV."""
    if not os.path.isfile(pairs_path):
        print(f"ERROR: pairs CSV not found: {pairs_path}", file=sys.stderr)
        sys.exit(1)

    pairs_df = pd.read_csv(pairs_path)
    print(f"  pairs: {len(pairs_df)} rows  ({pairs_path})")

    required_cols = AUDIO_COLS + VIDEO_COLS + TEXT_COLS + ["pair_id"]
    for col in required_cols:
        if col not in pairs_df.columns:
            print(f"ERROR: pairs CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    # Compute JSD scores for every row
    jsd_rows = []
    for i, row in pairs_df.iterrows():
        jsd_rows.append(compute_jsd_row(row))
        if (i + 1) % 1000 == 0:
            print(f"    scored {i + 1}/{len(pairs_df)} pairs...")

    jsd_df = pd.DataFrame(jsd_rows, columns=JSD_COLUMNS)
    out_df = pd.concat([pairs_df.reset_index(drop=True), jsd_df], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(out_df)} rows")

    # Summary stats
    print("\n  JSD summary:")
    for col in JSD_COLUMNS:
        print(f"    {col:18s}  mean={out_df[col].mean():.4f}  "
              f"max={out_df[col].max():.4f}")

    validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert row count, all JSD in [0,1], no NaN, pair_id preserved."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)}")

    for col in JSD_COLUMNS:
        assert col in df.columns, f"missing column: {col}"
        assert df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"

    assert df.isna().sum().sum() == 0, "NaN values present"
    assert df["pair_id"].is_unique, "pair_id values are not unique"

    print("  OK — JSD in [0,1], no NaN, pair_id unique")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute JSD scores (text/audio/video pairwise) for every pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        default="phase2_ravdess/pairs/incongruence_pairs.csv",
        help="Input pairs CSV from Harish Step 3",
    )
    parser.add_argument(
        "--out",
        default="phase2_ravdess/scores/pair_scores.csv",
        help="Output pair_scores CSV path",
    )
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START charan_jsd_scoring")
    run(args.pairs, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
