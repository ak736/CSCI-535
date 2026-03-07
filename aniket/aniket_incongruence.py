#!/usr/bin/env python3
"""
aniket_incongruence.py — Step 7 (Aniket)

Computes segment-level Text vs Audio Incongruence scores using
Jensen–Shannon (JS) divergence between P_text and P_audio.

Scoring formula:
    JS(P_text, P_audio) — symmetric, always finite, range [0, log(2)]
    Normalized to [0, 1] by dividing by log(2)

    confidence_weight = asr_confidence  (from segments CSV)
    final_score = js_score × confidence_weight

Both distributions operate over the SAME fixed label space:
    E = {happy, angry, sad, neutral}

Usage:
    python3 aniket/aniket_incongruence.py \
        --segments  outputs/segments/ES2002a_segments.csv \
        --text_emb  outputs/embeddings/ES2002a_text_emotions.csv \
        --audio_emb outputs/embeddings/ES2002a_audio_emotions.csv \
        --out       outputs/incongruence/ES2002a_scores.csv

Output columns (must match pipeline spec exactly):
    segment_id, start, end, speaker_id, transcript,
    incongruence_score, confidence_weight, final_score
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# JS Divergence
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
# Main
# ---------------------------------------------------------------------------

EMOTION_COLS = ["p_happy", "p_angry", "p_sad", "p_neutral"]


def run(
    segments_path: str,
    text_emb_path: str,
    audio_emb_path: str,
    out_path: str,
) -> None:
    # --- Validate inputs ---
    for path in (segments_path, text_emb_path, audio_emb_path):
        if not os.path.isfile(path):
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            sys.exit(1)

    segments_df  = pd.read_csv(segments_path)
    text_emb_df  = pd.read_csv(text_emb_path)
    audio_emb_df = pd.read_csv(audio_emb_path)

    # --- Column checks ---
    for col in ("segment_id", "start", "end", "speaker_id", "transcript", "asr_confidence"):
        if col not in segments_df.columns:
            print(f"ERROR: segments CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)
    for col in ["segment_id"] + EMOTION_COLS:
        if col not in text_emb_df.columns:
            print(f"ERROR: text_emotions CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)
        if col not in audio_emb_df.columns:
            print(f"ERROR: audio_emotions CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    print(f"  segments    : {len(segments_df)} rows")
    print(f"  text_emb    : {len(text_emb_df)} rows")
    print(f"  audio_emb   : {len(audio_emb_df)} rows")

    # --- Join all three on segment_id ---
    df = segments_df[["segment_id", "start", "end", "speaker_id", "transcript", "asr_confidence"]].copy()
    df = df.merge(
        text_emb_df[["segment_id"] + EMOTION_COLS].rename(
            columns={c: f"text_{c}" for c in EMOTION_COLS}
        ),
        on="segment_id", how="inner"
    )
    df = df.merge(
        audio_emb_df[["segment_id"] + EMOTION_COLS].rename(
            columns={c: f"audio_{c}" for c in EMOTION_COLS}
        ),
        on="segment_id", how="inner"
    )

    if len(df) < len(segments_df):
        print(f"  WARNING: {len(segments_df) - len(df)} segments dropped during join (segment_id mismatch)")

    # --- Compute JS divergence and confidence weight ---
    js_scores   = []
    conf_weights = []
    final_scores = []

    for _, row in df.iterrows():
        p_text  = np.array([row[f"text_{c}"]  for c in EMOTION_COLS], dtype=np.float64)
        p_audio = np.array([row[f"audio_{c}"] for c in EMOTION_COLS], dtype=np.float64)

        # Normalize (guard against float drift from upstream models)
        p_text  = p_text  / p_text.sum()  if p_text.sum()  > 0 else np.ones(4) / 4
        p_audio = p_audio / p_audio.sum() if p_audio.sum() > 0 else np.ones(4) / 4

        js  = js_divergence(p_text, p_audio)
        cw  = float(row["asr_confidence"])        # confidence_weight = asr_confidence (Phase 1)
        cw  = max(0.0, min(1.0, cw))              # clamp to [0, 1]
        fs  = round(js * cw, 6)

        js_scores.append(round(js, 6))
        conf_weights.append(round(cw, 6))
        final_scores.append(fs)

    df["incongruence_score"]  = js_scores
    df["confidence_weight"]   = conf_weights
    df["final_score"]         = final_scores

    # --- Write output with exact column order ---
    out_cols = [
        "segment_id", "start", "end", "speaker_id", "transcript",
        "incongruence_score", "confidence_weight", "final_score"
    ]
    out_df = df[out_cols].copy()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(out_df)} rows")

    # --- Summary stats ---
    print(f"\n  Incongruence score summary:")
    print(f"    mean  final_score : {out_df['final_score'].mean():.4f}")
    print(f"    max   final_score : {out_df['final_score'].max():.4f}")
    print(f"    top 3 incongruent segments:")
    top3 = out_df.nlargest(3, "final_score")[["segment_id", "speaker_id", "transcript", "final_score"]]
    for _, r in top3.iterrows():
        print(f"      seg {int(r['segment_id']):4d} [{r['speaker_id']}]  score={r['final_score']:.4f}  \"{r['transcript'][:60]}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute JS divergence incongruence scores (Text vs Audio)"
    )
    parser.add_argument("--segments",  required=True, help="Path to segments CSV")
    parser.add_argument("--text_emb",  required=True, help="Path to text_emotions CSV")
    parser.add_argument("--audio_emb", required=True, help="Path to audio_emotions CSV")
    parser.add_argument("--out",       required=True, help="Path to output scores CSV")
    args = parser.parse_args()

    print(f"START incongruence scoring")
    run(args.segments, args.text_emb, args.audio_emb, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
