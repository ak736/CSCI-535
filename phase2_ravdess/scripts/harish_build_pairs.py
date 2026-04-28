#!/usr/bin/env python3
"""
harish_build_pairs.py — Phase 2 Step 3 (Harish)

Merge the three emotion CSVs and construct synthetic congruent/incongruent
pairs by mixing audio from one emotion with video from another (for the same
actor, statement, intensity, repetition).

    Congruent pair:   audio(happy) + video(happy)   → label = 0
    Incongruent pair: audio(happy) + video(angry)   → label = 1

INPUTS:
    phase2_ravdess/metadata/ravdess_metadata.csv
    phase2_ravdess/emotions/audio_emotions.csv
    phase2_ravdess/emotions/video_emotions.csv
    phase2_ravdess/emotions/text_emotions.csv

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
                  audio_tag: str, video_tag: str) -> str:
    """
    pair_id format: {actor:02d}_{stmt}_{int}_{rep}_A{audio_tag}_V{video_tag}

    audio_tag / video_tag use metadata ``emotion_label`` (neutral, calm, happy,
    …) so ids stay unique when several emotion_codes share the same 4-class.
    """
    return f"{actor:02d}_{stmt}_{intensity}_{rep}_A{audio_tag}_V{video_tag}"


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

    required_emo_cols = {"clip_id", "actor", "emotion_code", "emotion_4class", *EMOTION_COLS}
    for name, df in [("audio", audio_df), ("video", video_df)]:
        missing = required_emo_cols - set(df.columns)
        if missing:
            print(f"ERROR: {name} CSV missing columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

    for col in ("statement", "sentence", *EMOTION_COLS):
        if col not in text_df.columns:
            print(f"ERROR: text CSV missing column: {col}", file=sys.stderr)
            sys.exit(1)

    meta_join_cols = [
        "clip_id", "gender", "statement", "intensity", "repetition", "emotion_label",
    ]
    missing_meta = set(meta_join_cols) - set(metadata_df.columns)
    if missing_meta:
        print(f"ERROR: metadata CSV missing columns: {sorted(missing_meta)}", file=sys.stderr)
        sys.exit(1)

    meta_join = metadata_df[meta_join_cols].drop_duplicates(subset=["clip_id"], keep="first")

    audio_full = audio_df.merge(meta_join, on="clip_id", how="inner")
    video_full = video_df.merge(meta_join, on="clip_id", how="inner")

    text_by_stmt: dict[int, dict[str, float]] = {}
    for _, tr in text_df.iterrows():
        sid = int(tr["statement"])
        text_by_stmt[sid] = {f"p_text_{c.split('_', 1)[1]}": float(tr[c]) for c in EMOTION_COLS}

    if set(text_by_stmt.keys()) != {1, 2}:
        print(
            f"ERROR: text_emotions.csv must have statement 1 and 2, got {sorted(text_by_stmt)}",
            file=sys.stderr,
        )
        sys.exit(1)

    group_cols = ["actor", "statement", "intensity", "repetition"]
    audio_groups = dict(tuple(audio_full.groupby(group_cols, sort=True)))
    video_groups = dict(tuple(video_full.groupby(group_cols, sort=True)))

    rows: list[dict] = []
    for key in sorted(audio_groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3])):
        if key not in video_groups:
            continue
        a_g = audio_groups[key]
        v_g = video_groups[key]
        actor, stmt, intensity, repetition = key
        gender = str(a_g.iloc[0]["gender"])
        text_probs = text_by_stmt[int(stmt)]

        for _, a_row in a_g.iterrows():
            for _, v_row in v_g.iterrows():
                pair_id = build_pair_id(
                    int(actor),
                    int(stmt),
                    int(intensity),
                    int(repetition),
                    str(a_row["emotion_label"]),
                    str(v_row["emotion_label"]),
                )
                label = 0 if a_row["emotion_4class"] == v_row["emotion_4class"] else 1
                rows.append(
                    {
                        "pair_id": pair_id,
                        "actor": int(actor),
                        "gender": gender,
                        "statement": int(stmt),
                        "intensity": int(intensity),
                        "repetition": int(repetition),
                        "audio_clip_id": str(a_row["clip_id"]),
                        "video_clip_id": str(v_row["clip_id"]),
                        "audio_emotion_4class": str(a_row["emotion_4class"]),
                        "video_emotion_4class": str(v_row["emotion_4class"]),
                        "label": label,
                        "p_audio_happy": float(a_row["p_happy"]),
                        "p_audio_angry": float(a_row["p_angry"]),
                        "p_audio_sad": float(a_row["p_sad"]),
                        "p_audio_neutral": float(a_row["p_neutral"]),
                        "p_video_happy": float(v_row["p_happy"]),
                        "p_video_angry": float(v_row["p_angry"]),
                        "p_video_sad": float(v_row["p_sad"]),
                        "p_video_neutral": float(v_row["p_neutral"]),
                        **text_probs,
                    }
                )

    if not rows:
        print("ERROR: no pairs generated (check audio/video CSVs and metadata)", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.sort_values("pair_id", kind="stable").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(df)} rows")
    print(f"  label distribution: {df['label'].value_counts().to_dict()}")
    validate(out_path)


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
