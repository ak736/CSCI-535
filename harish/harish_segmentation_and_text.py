#!/usr/bin/env python3
"""
harish_segmentation_and_text.py — Segmentation + Text Emotion Embeddings (Harish)

PART 1: Build 3s windows (1.5s hop) from merged CSV; single-speaker only, >= 3 words.
PART 2: For each segment transcript, compute P_text = [p_happy, p_angry, p_sad, p_neutral]
        via HuggingFace emotion model (j-hartmann/emotion-english-distilroberta-base).

Usage (both parts, default):
    python harish/harish_segmentation_and_text.py \
        --merged outputs/merged/ES2002a_merged.csv \
        --part ES2002a \
        --out_dir outputs

Usage (Part 1 only — write segments CSV):
    python harish/harish_segmentation_and_text.py \
        --merged outputs/merged/ES2002a_merged.csv \
        --part ES2002a \
        --out_dir outputs \
        --segments_only

Usage (Part 2 only — requires existing segments CSV):
    python harish/harish_segmentation_and_text.py \
        --segments outputs/segments/ES2002a_segments.csv \
        --part ES2002a \
        --out_dir outputs \
        --embeddings_only

Outputs:
    {out_dir}/segments/{part}_segments.csv       — segment_id, start, end, speaker_id, transcript, asr_confidence
    {out_dir}/embeddings/{part}_text_emotions.csv — segment_id, p_happy, p_angry, p_sad, p_neutral
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Part 1: Fixed-length segmentation (3s window, 1.5s hop)
# ---------------------------------------------------------------------------

WINDOW_SEC = 3.0
HOP_SEC = 1.5
MIN_WORDS = 3


def _build_word_timeline(merged_df: pd.DataFrame) -> list[tuple[float, float, str, str, float]]:
    """(start, end, word, speaker_id, asr_confidence) per word; words get proportional time within segment."""
    timeline = []
    for _, row in merged_df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        speaker = row["speaker_id"]
        text = (row.get("transcript") or "").strip()
        conf = float(row.get("asr_confidence", 1.0))
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        duration = end - start
        step = duration / len(words)
        for i, w in enumerate(words):
            ws = start + i * step
            we = start + (i + 1) * step
            timeline.append((ws, we, w, speaker, conf))
    return sorted(timeline, key=lambda x: x[0])


def _speakers_in_window(start: float, end: float, merged_df: pd.DataFrame) -> set:
    speakers = set()
    for _, row in merged_df.iterrows():
        s, e = float(row["start"]), float(row["end"])
        if e > start and s < end:
            speakers.add(row["speaker_id"])
    return speakers


def _words_in_window(start: float, end: float, timeline: list) -> list[tuple[float, float, str, str, float]]:
    """Words whose midpoint lies in [start, end]."""
    return [
        item for item in timeline
        if start <= (item[0] + item[1]) / 2 <= end
    ]


def _transcript_and_confidence(
    start: float, end: float,
    timeline: list,
    merged_df: pd.DataFrame,
) -> tuple[str, float]:
    """Transcript = words in window joined; asr_confidence = mean of overlapping segments' asr_confidence."""
    words_in = _words_in_window(start, end, timeline)
    transcript = " ".join(w[2] for w in words_in).strip()

    # Overlapping merged rows' asr_confidence → mean
    overlapping = merged_df[
        (merged_df["end"] > start) & (merged_df["start"] < end)
    ]
    asr_confidence = float(overlapping["asr_confidence"].mean()) if len(overlapping) else 0.0
    asr_confidence = round(asr_confidence, 4)
    return transcript, asr_confidence


def _primary_speaker(start: float, end: float, timeline: list) -> str | None:
    """Speaker with most word-time in [start, end]."""
    speaker_time = []
    for ws, we, _, sp, _ in timeline:
        overlap = max(0.0, min(end, we) - max(start, ws))
        if overlap > 0:
            speaker_time.append((sp, overlap))
    if not speaker_time:
        return None
    total = {}
    for sp, t in speaker_time:
        total[sp] = total.get(sp, 0.0) + t
    return max(total, key=total.get)


def run_part1_segmentation(merged_path: str, segments_path: str) -> None:
    """Part 1: Create 3s windows, single-speaker, >= 3 words; write segments CSV."""
    if not os.path.isfile(merged_path):
        print(f"ERROR: merged CSV not found: {merged_path}", file=sys.stderr)
        sys.exit(1)

    merged_df = pd.read_csv(merged_path)
    for col in ("segment_id", "start", "end", "speaker_id", "transcript", "asr_confidence"):
        if col not in merged_df.columns:
            print(f"ERROR: merged CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    timeline = _build_word_timeline(merged_df)
    audio_end = float(merged_df["end"].max())
    rows = []
    segment_id = 0
    t = 0.0

    while t + WINDOW_SEC <= audio_end:
        start = t
        end = t + WINDOW_SEC

        speakers = _speakers_in_window(start, end, merged_df)
        if len(speakers) != 1:
            t += HOP_SEC
            continue

        words_in = _words_in_window(start, end, timeline)
        if len(words_in) < MIN_WORDS:
            t += HOP_SEC
            continue

        transcript, asr_confidence = _transcript_and_confidence(start, end, timeline, merged_df)
        primary = _primary_speaker(start, end, timeline)
        speaker_id = primary if primary else list(speakers)[0]

        rows.append({
            "segment_id": segment_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker_id": speaker_id,
            "transcript": transcript,
            "asr_confidence": asr_confidence,
        })
        segment_id += 1
        t += HOP_SEC

    os.makedirs(os.path.dirname(os.path.abspath(segments_path)), exist_ok=True)
    out_df = pd.DataFrame(rows, columns=["segment_id", "start", "end", "speaker_id", "transcript", "asr_confidence"])
    out_df.to_csv(segments_path, index=False)
    print(f"WROTE {segments_path} {len(out_df)} rows (Part 1)")


# ---------------------------------------------------------------------------
# Part 2: Text emotion embeddings (HuggingFace → P_text)
# ---------------------------------------------------------------------------

# j-hartmann/emotion-english-distilroberta-base labels: anger, disgust, fear, joy, neutral, sadness, surprise
LABEL_TO_FOUR = {
    "joy":      "happy",
    "anger":    "angry",
    "sadness":  "sad",
    "neutral":  "neutral",
    "disgust":  "neutral",  # map extra classes into neutral
    "fear":     "neutral",
    "surprise": "neutral",
}
OUTPUT_LABELS = ["p_happy", "p_angry", "p_sad", "p_neutral"]


def run_part2_embeddings(segments_path: str, embeddings_path: str) -> None:
    """Part 2: For each segment transcript, compute P_text; write embeddings CSV."""
    if not os.path.isfile(segments_path):
        print(f"ERROR: segments CSV not found: {segments_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: install transformers: pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    segments_df = pd.read_csv(segments_path)
    for col in ("segment_id", "transcript"):
        if col not in segments_df.columns:
            print(f"ERROR: segments CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    print("Loading HuggingFace emotion model: j-hartmann/emotion-english-distilroberta-base")
    pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,  # return all labels with scores
    )

    rows = []
    for _, row in segments_df.iterrows():
        seg_id = int(row["segment_id"])
        text = (row.get("transcript") or "").strip()
        # Empty transcript → uniform distribution so probs sum to 1
        if not text:
            probs = {k: 0.25 for k in OUTPUT_LABELS}
        else:
            out = pipe(text, truncation=True, max_length=512)
            # With top_k=None, pipeline returns list-of-lists for batch: one list per input.
            # For a single string we get out = [[{label, score}, ...]]; take out[0].
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
            probs = {
                "p_happy":   round(four_probs["happy"],   6),
                "p_angry":   round(four_probs["angry"],   6),
                "p_sad":     round(four_probs["sad"],     6),
                "p_neutral": round(four_probs["neutral"], 6),
            }
        rows.append({"segment_id": seg_id, **probs})

    out_df = pd.DataFrame(rows, columns=["segment_id"] + OUTPUT_LABELS)
    os.makedirs(os.path.dirname(os.path.abspath(embeddings_path)), exist_ok=True)
    out_df.to_csv(embeddings_path, index=False)
    print(f"WROTE {embeddings_path} {len(out_df)} rows (Part 2)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentation (3s windows) + Text emotion embeddings (Harish)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--merged", type=str, help="Path to merged CSV (for Part 1)")
    parser.add_argument("--segments", type=str, help="Path to segments CSV (for Part 2 only)")
    parser.add_argument("--part", type=str, required=True, help="Meeting part id (e.g. ES2002a)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--segments_only", action="store_true", help="Run only Part 1 (write segments CSV)")
    parser.add_argument("--embeddings_only", action="store_true", help="Run only Part 2 (requires --segments)")

    args = parser.parse_args()

    segments_path = os.path.join(args.out_dir, "segments", f"{args.part}_segments.csv")
    embeddings_path = os.path.join(args.out_dir, "embeddings", f"{args.part}_text_emotions.csv")

    if args.embeddings_only:
        if not args.segments:
            print("ERROR: --embeddings_only requires --segments", file=sys.stderr)
            sys.exit(1)
        run_part2_embeddings(args.segments, embeddings_path)
        return

    if args.segments_only:
        if not args.merged:
            print("ERROR: --segments_only requires --merged", file=sys.stderr)
            sys.exit(1)
        run_part1_segmentation(args.merged, segments_path)
        return

    # Default: run both Part 1 and Part 2
    if not args.merged:
        print("ERROR: default run requires --merged", file=sys.stderr)
        sys.exit(1)
    run_part1_segmentation(args.merged, segments_path)
    run_part2_embeddings(segments_path, embeddings_path)
    print("DONE (Part 1 + Part 2)")


if __name__ == "__main__":
    main()
