#!/usr/bin/env python3
"""
harish_text_emotion.py — Phase 2 Step 2c (Harish)

RAVDESS actors read only 2 fixed sentences. We therefore only need to
predict emotion probabilities on those 2 sentences — not on every clip.

    Statement 1: "Kids are talking by the door"
    Statement 2: "Dogs are sitting by the door"

MODEL: j-hartmann/emotion-english-distilroberta-base

The model returns 7 labels (joy, anger, sadness, neutral, disgust, fear,
surprise). We collapse them to our 4-class space via LABEL_TO_FOUR, then
renormalize.

INPUT:
    (none — sentences are hardcoded)

OUTPUT:
    phase2_ravdess/emotions/text_emotions.csv

    Columns: statement (int 1|2), sentence (str), p_happy, p_angry, p_sad, p_neutral
    Exactly 2 rows. Probabilities sum to 1.0 per row.

RUN:
    python3 phase2_ravdess/scripts/harish_text_emotion.py \\
        --out phase2_ravdess/emotions/text_emotions.csv
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

TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# 7 model labels → our 4-class space (copied from Phase 1)
LABEL_TO_FOUR = {
    "joy":      "happy",
    "anger":    "angry",
    "sadness":  "sad",
    "neutral":  "neutral",
    "disgust":  "neutral",
    "fear":     "neutral",
    "surprise": "neutral",
}

OUTPUT_LABELS = ["p_happy", "p_angry", "p_sad", "p_neutral"]

SENTENCES = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

EXPECTED_ROWS = 2

OUTPUT_COLUMNS = ["statement", "sentence"] + OUTPUT_LABELS


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_text_pipeline():
    """Load the distilroberta emotion classification pipeline."""
    try:
        from transformers import pipeline
    except ImportError as e:
        print(
            "ERROR: could not import transformers.\n"
            f"  Python: {sys.executable}\n"
            f"  Reason: {e}\n"
            "  Fix: use the SAME interpreter you run this script with, e.g.\n"
            "    python3 -m pip install transformers torch",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loading text emotion model: {TEXT_MODEL_ID}")
    pipe = pipeline(
        "text-classification",
        model=TEXT_MODEL_ID,
        top_k=None,  # return all 7 label scores
    )
    return pipe


def predict_text_probs(pipe, sentence: str) -> dict:
    """
    Run the pipeline on a sentence, collapse 7 labels → 4-class, renormalize.
    Returns dict with keys p_happy, p_angry, p_sad, p_neutral (sum to 1.0).
    """
    if not (sentence or "").strip():
        return {k: 0.25 for k in OUTPUT_LABELS}

    out = pipe(sentence, truncation=True, max_length=512)
    if out and isinstance(out[0], list):
        label_list = out[0]
    else:
        label_list = out

    four_probs = {"happy": 0.0, "angry": 0.0, "sad": 0.0, "neutral": 0.0}
    for item in label_list:
        lab = item["label"].strip().lower()
        score = float(item["score"])
        target = LABEL_TO_FOUR.get(lab, "neutral")
        four_probs[target] += score

    total = sum(four_probs.values())
    if total <= 0:
        four_probs = {"happy": 0.25, "angry": 0.25, "sad": 0.25, "neutral": 0.25}
    else:
        four_probs = {k: v / total for k, v in four_probs.items()}

    return {
        "p_happy": round(four_probs["happy"], 6),
        "p_angry": round(four_probs["angry"], 6),
        "p_sad": round(four_probs["sad"], 6),
        "p_neutral": round(four_probs["neutral"], 6),
    }


def run(out_path: str) -> None:
    """Score the 2 fixed sentences and write text_emotions.csv."""
    pipe = load_text_pipeline()

    rows = []
    for stmt_id, sentence in SENTENCES.items():
        probs = predict_text_probs(pipe, sentence)
        rows.append({"statement": stmt_id, "sentence": sentence, **probs})

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(df)} rows")
    validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert 2 rows, probs sum to ~1.0, no NaN."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)} (expected {EXPECTED_ROWS})")

    assert len(df) == EXPECTED_ROWS, f"row count {len(df)} != {EXPECTED_ROWS}"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert list(df.columns) == OUTPUT_COLUMNS, f"column mismatch: {list(df.columns)}"

    sums = df[OUTPUT_LABELS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), (
        f"probabilities do not sum to 1.0: {sums.tolist()}"
    )

    print("  OK — 2 rows, probs sum to 1.0, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the 2 fixed RAVDESS sentences with distilroberta emotion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="phase2_ravdess/emotions/text_emotions.csv",
        help="Output text_emotions CSV path",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate an existing text_emotions CSV and exit",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START harish_text_emotion")
    run(args.out)
    print("DONE")


if __name__ == "__main__":
    main()
