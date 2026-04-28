#!/usr/bin/env python3
"""
aaditya_video_emotion.py — Phase 2 Step 2b (Aaditya)

Map pre-extracted OpenFace AU features to emotion probabilities.

NOTE — NO py-feat, NO OpenFace execution needed. The RAVDESS Facial
Landmark Tracking dataset ships per-trial tracking CSVs with frame-level
AU intensities (OpenFace 2.1.0 output). We just read them.

FACS mapping:
    happy:   AU06 (cheek raiser) + AU12 (lip corner puller)
    angry:   AU04 + AU17 + AU25
    sad:     AU01 + AU04 + AU15
    neutral: inverse of total AU activation

INPUTS:
    phase2_ravdess/metadata/ravdess_metadata.csv
    Datasets/ravdess/facial_tracking/*.csv  (pre-extracted, also Actor_XX/ layout)

FILENAME MATCHING:
    Audio wav    = 03-01-05-02-01-01-08.wav    (modality=03 audio-only)
    Tracking csv = 01-01-05-02-01-01-08.csv    (modality=01 full AV)
    → share the same clip_id. For each metadata row, swap leading "03" → "01"
      on the basename to find the tracking CSV.

OUTPUT:
    phase2_ravdess/emotions/video_emotions.csv

    Columns:
        clip_id, actor, emotion_code, emotion_4class,
        p_happy, p_angry, p_sad, p_neutral

RUN:
    python3 phase2_ravdess/scripts/aaditya_video_emotion.py \\
        --metadata     phase2_ravdess/metadata/ravdess_metadata.csv \\
        --tracking_root Datasets/ravdess/facial_tracking \\
        --out          phase2_ravdess/emotions/video_emotions.csv
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

# OpenFace AU_r (regression / intensity) columns we care about
AU_COLS_OPENFACE = [
    "AU01_r", "AU02_r", "AU04_r", "AU06_r",
    "AU12_r", "AU15_r", "AU17_r", "AU25_r",
]
# Short names we use internally (strip the _r suffix)
AU_COLS_OURS = [c.replace("_r", "") for c in AU_COLS_OPENFACE]

# OpenFace quality columns — filter to valid frames only
QUALITY_COLS = ["confidence", "success"]
CONFIDENCE_THRESHOLD = 0.5

OUTPUT_LABELS = ["p_happy", "p_angry", "p_sad", "p_neutral"]

OUTPUT_COLUMNS = [
    "clip_id", "actor", "emotion_code", "emotion_4class",
] + OUTPUT_LABELS

# Uniform distribution used when a tracking CSV is missing or has no valid frames
UNIFORM_PROBS = {"p_happy": 0.25, "p_angry": 0.25, "p_sad": 0.25, "p_neutral": 0.25}


# ---------------------------------------------------------------------------
# FACS mapping
# ---------------------------------------------------------------------------

def au_to_emotion_scores(row: dict) -> tuple[float, float, float, float]:
    """Map AU intensities to raw emotion scores.

    Based on FACS literature:
    - Happy: AU06 + AU12 (Duchenne smile)
    - Angry: AU04 + AU17 + AU25
    - Sad:   AU01 + AU04 + AU15
    - Neutral: inverse of total AU activation
    """
    au01 = float(row.get("AU01", 0))
    au02 = float(row.get("AU02", 0))
    au04 = float(row.get("AU04", 0))
    au06 = float(row.get("AU06", 0))
    au12 = float(row.get("AU12", 0))
    au15 = float(row.get("AU15", 0))
    au17 = float(row.get("AU17", 0))
    au25 = float(row.get("AU25", 0))

    happy_score = 0.5 * au06 + 0.5 * au12
    angry_score = 0.4 * au04 + 0.3 * au17 + 0.3 * au25
    sad_score   = 0.4 * au01 + 0.3 * au04 + 0.3 * au15

    total_activation = au01 + au02 + au04 + au06 + au12 + au15 + au17 + au25
    neutral_score = max(0.0, 1.0 - 0.15 * total_activation)

    return happy_score, angry_score, sad_score, neutral_score


def softmax(scores) -> np.ndarray:
    """Numerically stable softmax."""
    scores = np.array(scores, dtype=np.float64)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    return probs


# ---------------------------------------------------------------------------
# RAVDESS-specific tracking CSV loading
# ---------------------------------------------------------------------------

def audio_filename_to_tracking_filename(audio_filename: str) -> str:
    """
    RAVDESS audio: '03-01-05-02-01-01-08.wav'  (modality=03)
    Tracking csv:  '01-01-05-02-01-01-08.csv'  (modality=01)
    """
    base = os.path.splitext(audio_filename)[0]
    parts = base.split("-")
    assert len(parts) == 7, f"Unexpected filename: {audio_filename}"
    parts[0] = "01"  # swap modality code to tracking's "01"
    return "-".join(parts) + ".csv"


def mean_pool_valid_frames(csv_path: str) -> dict | None:
    """
    Load a tracking CSV, filter to confidence > 0.5 frames, and additionally
    require success == 1 if that column exists. Mean-pool AU*_r columns.
    Returns dict of mean AU values, or None if empty.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  WARNING: failed to read {csv_path}: {e}")
        return None

    # Strip whitespace from column names (OpenFace CSVs sometimes have " AU01_r")
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns are present (success is optional in some releases)
    missing = [c for c in AU_COLS_OPENFACE + ["confidence"] if c not in df.columns]
    if missing:
        print(f"  WARNING: {csv_path} missing columns {missing}")
        return None

    # Filter valid frames — use success column only if present
    mask = df["confidence"] > CONFIDENCE_THRESHOLD
    if "success" in df.columns:
        mask = mask & (df["success"] == 1)
    valid = df[mask]
    if len(valid) == 0:
        return None

    means = {}
    for openface_col, our_col in zip(AU_COLS_OPENFACE, AU_COLS_OURS):
        means[our_col] = float(valid[openface_col].mean())
    return means


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(metadata_path: str, tracking_root: str, out_path: str) -> None:
    """For every clip_id in metadata, load tracking CSV → AU pool → softmax → probs."""
    if not os.path.isfile(metadata_path):
        print(f"ERROR: metadata CSV not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(tracking_root):
        print(f"ERROR: tracking_root not found: {tracking_root}", file=sys.stderr)
        sys.exit(1)

    metadata_df = pd.read_csv(metadata_path)
    print(f"  metadata: {len(metadata_df)} rows  ({metadata_path})")

    rows = []
    n_missing = 0
    n_uniform = 0

    for _, meta in metadata_df.iterrows():
        clip_id  = meta["clip_id"]
        actor    = int(meta["actor"])
        emo_code = int(meta["emotion_code"])
        emo_4    = meta["emotion_4class"]
        audio_fn = meta["filename"]

        tracking_fn   = audio_filename_to_tracking_filename(audio_fn)
        # Try Actor_XX/ subdirectory first; fall back to flat layout
        tracking_path_subdir = os.path.join(tracking_root, f"Actor_{actor:02d}", tracking_fn)
        tracking_path_flat   = os.path.join(tracking_root, tracking_fn)
        if os.path.isfile(tracking_path_subdir):
            tracking_path = tracking_path_subdir
        else:
            tracking_path = tracking_path_flat

        probs = None
        if os.path.isfile(tracking_path):
            means = mean_pool_valid_frames(tracking_path)
            if means is not None:
                h, a, s, n = au_to_emotion_scores(means)
                sm = softmax([h, a, s, n])
                sm = sm / sm.sum()  # renormalize to sum to 1.0
                probs = {
                    "p_happy":   round(float(sm[0]), 6),
                    "p_angry":   round(float(sm[1]), 6),
                    "p_sad":     round(float(sm[2]), 6),
                    "p_neutral": round(float(sm[3]), 6),
                }
        else:
            n_missing += 1
            if n_missing <= 5:
                print(f"  MISSING tracking: {tracking_path}")

        if probs is None:
            probs = dict(UNIFORM_PROBS)
            n_uniform += 1
            print(f"  UNIFORM fallback: {clip_id}")

        rows.append({
            "clip_id": clip_id,
            "actor": actor,
            "emotion_code": emo_code,
            "emotion_4class": emo_4,
            **probs,
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(df)} rows")
    print(f"  missing tracking files: {n_missing}")
    print(f"  uniform-fallback rows : {n_uniform}")
    validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert rows match metadata, probs sum to ~1.0, no NaN, unique clip_ids."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)}")

    assert df["clip_id"].is_unique, "clip_id values are not unique"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert list(df.columns) == OUTPUT_COLUMNS, f"column mismatch: {list(df.columns)}"

    sums = df[OUTPUT_LABELS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "probs do not sum to 1.0"

    print("  OK — unique clip_ids, probs sum to 1.0, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAVDESS video emotion prediction from pre-extracted OpenFace tracking CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        default="phase2_ravdess/metadata/ravdess_metadata.csv",
        help="Input metadata CSV from Harish Step 1",
    )
    parser.add_argument(
        "--tracking_root",
        default="Datasets/ravdess/facial_tracking",
        help="Root directory of RAVDESS facial tracking CSVs (Actor_01 ... Actor_24)",
    )
    parser.add_argument(
        "--out",
        default="phase2_ravdess/emotions/video_emotions.csv",
        help="Output video_emotions CSV path",
    )
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START aaditya_video_emotion")
    run(args.metadata, args.tracking_root, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
