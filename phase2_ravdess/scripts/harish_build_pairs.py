#!/usr/bin/env python3
"""
harish_build_pairs.py — Phase 2 Step 3 (Harish)

OWNER: Harish
STEP : 3 (Day 4) — WAITS for Charan Step 2a + Aaditya Step 2b + Harish Step 2c

Merge the three emotion CSVs and construct synthetic congruent/incongruent
pairs by mixing audio from one emotion with video from another (for the same
actor, statement, intensity, repetition).

    Congruent pair:   audio(happy) + video(happy)   → label = 0
    Incongruent pair: audio(happy) + video(angry)   → label = 1

INPUTS:
    phase2_ravdess/metadata/ravdess_metadata.csv          (from Step 1)
    phase2_ravdess/emotions/audio_emotions.csv            (from Charan Step 2a)
    phase2_ravdess/emotions/video_emotions.csv            (from Aaditya Step 2b)
    phase2_ravdess/emotions/text_emotions.csv             (from Harish Step 2c)

OUTPUT:
    phase2_ravdess/pairs/incongruence_pairs.csv

    Columns:
        pair_id, actor, gender, statement, intensity, repetition,
        audio_clip_id, video_clip_id,
        audio_emotion_4class, video_emotion_4class, label,
        p_audio_happy, p_audio_angry, p_audio_sad, p_audio_neutral,
        p_video_happy, p_video_angry, p_video_sad, p_video_neutral,
        p_text_happy,  p_text_angry,  p_text_sad,  p_text_neutral

RUN:
    python3 phase2_ravdess/scripts/harish_build_pairs.py \\
        --metadata phase2_ravdess/metadata/ravdess_metadata.csv \\
        --audio    phase2_ravdess/emotions/audio_emotions.csv \\
        --video    phase2_ravdess/emotions/video_emotions.csv \\
        --text     phase2_ravdess/emotions/text_emotions.csv \\
        --out      phase2_ravdess/pairs/incongruence_pairs.csv

DOWNSTREAM CONSUMER: Charan Step 4 (JSD scoring)
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

EMOTION_COLS = ["p_happy", "p_angry", "p_sad", "p_neutral"]
FOUR_CLASS_SET = {"happy", "angry", "sad", "neutral"}

OUTPUT_COLUMNS = [
    "pair_id", "actor", "gender", "statement", "intensity", "repetition",
    "audio_clip_id", "video_clip_id",
    "audio_emotion_4class", "video_emotion_4class", "label",
    "p_audio_happy", "p_audio_angry", "p_audio_sad", "p_audio_neutral",
    "p_video_happy", "p_video_angry", "p_video_sad", "p_video_neutral",
    "p_text_happy",  "p_text_angry",  "p_text_sad",  "p_text_neutral",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_pair_id(actor: int, stmt: int, intensity: int, rep: int,
                  audio_emo: str, video_emo: str) -> str:
    """
    pair_id format: {actor}_{stmt}_{int}_{rep}_A{audio_emo}_V{video_emo}
    Example: 08_1_2_1_Aangry_Vhappy
    """
    return f"{actor:02d}_{stmt}_{intensity}_{rep}_A{audio_emo}_V{video_emo}"


def run(metadata_path: str, audio_path: str, video_path: str,
        text_path: str, out_path: str) -> None:
    """Join all 4 inputs and enumerate congruent/incongruent pairs."""

    # --- Validate inputs exist ---
    for label, path in [("metadata", metadata_path), ("audio", audio_path),
                        ("video", video_path), ("text", text_path)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    metadata_df = pd.read_csv(metadata_path)
    audio_df    = pd.read_csv(audio_path)
    video_df    = pd.read_csv(video_path)
    text_df     = pd.read_csv(text_path)

    print(f"  metadata : {len(metadata_df)} rows")
    print(f"  audio_emo: {len(audio_df)} rows")
    print(f"  video_emo: {len(video_df)} rows")
    print(f"  text_emo : {len(text_df)} rows")

    # TODO(Harish): implement pair construction
    #
    # PSEUDOCODE:
    #   1. Merge metadata with audio_df on clip_id → adds audio probs + gender + emotion_4class.
    #   2. Merge metadata with video_df on clip_id → adds video probs + emotion_4class.
    #      (Video may be missing rows; inner-join is fine — downstream expects both present.)
    #   3. Build text lookup: dict[statement_int] → (p_happy, p_angry, p_sad, p_neutral).
    #   4. Group audio rows and video rows by (actor, statement, intensity, repetition).
    #      Within each group, Cartesian-join audio emotions x video emotions.
    #      Skip any combo where either side is missing.
    #   5. For each (audio_row, video_row) combination:
    #        - label = 0 if audio.emotion_4class == video.emotion_4class else 1
    #        - pair_id = build_pair_id(actor, stmt, intensity, rep, audio_emo, video_emo)
    #        - Pull p_audio_* from audio_row, p_video_* from video_row, p_text_* via lookup.
    #        - Append dict to rows list.
    #   6. Build DataFrame with OUTPUT_COLUMNS order.
    #   7. Print label distribution (0 vs 1) and write CSV.
    print("TODO: implement pair construction loop")
    sys.exit(0)

    # df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # df.to_csv(out_path, index=False)
    # print(f"WROTE {out_path} {len(df)} rows")
    # print(f"  label distribution: {df['label'].value_counts().to_dict()}")
    # validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert pair_id unique, label logic, probs sum to ~1.0, no NaN."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)}")

    assert df["pair_id"].is_unique, "pair_id values are not unique"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert list(df.columns) == OUTPUT_COLUMNS, (
        f"column mismatch: {list(df.columns)}"
    )

    # label logic: 0 iff audio == video emotion, 1 iff different
    same = (df["audio_emotion_4class"] == df["video_emotion_4class"])
    assert ((df["label"] == 0) == same).all(), (
        "label does not match audio_emotion_4class == video_emotion_4class"
    )

    # audio probs sum to ~1, video probs sum to ~1, text probs sum to ~1
    audio_sum = df[[f"p_audio_{e.split('_')[1]}" for e in EMOTION_COLS]].sum(axis=1)
    video_sum = df[[f"p_video_{e.split('_')[1]}" for e in EMOTION_COLS]].sum(axis=1)
    text_sum  = df[[f"p_text_{e.split('_')[1]}"  for e in EMOTION_COLS]].sum(axis=1)

    assert np.allclose(audio_sum, 1.0, atol=1e-3), "audio probs do not sum to 1.0"
    assert np.allclose(video_sum, 1.0, atol=1e-3), "video probs do not sum to 1.0"
    assert np.allclose(text_sum,  1.0, atol=1e-3), "text probs do not sum to 1.0"

    # 4-class labels only
    assert set(df["audio_emotion_4class"].unique()).issubset(FOUR_CLASS_SET)
    assert set(df["video_emotion_4class"].unique()).issubset(FOUR_CLASS_SET)

    print(f"  label distribution: {df['label'].value_counts().to_dict()}")
    print("  OK — unique pair_ids, consistent label logic, probs sum to 1.0, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build synthetic congruent/incongruent pairs from RAVDESS emotion CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", default="phase2_ravdess/metadata/ravdess_metadata.csv")
    parser.add_argument("--audio",    default="phase2_ravdess/emotions/audio_emotions.csv")
    parser.add_argument("--video",    default="phase2_ravdess/emotions/video_emotions.csv")
    parser.add_argument("--text",     default="phase2_ravdess/emotions/text_emotions.csv")
    parser.add_argument("--out",      default="phase2_ravdess/pairs/incongruence_pairs.csv")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START harish_build_pairs")
    run(args.metadata, args.audio, args.video, args.text, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
