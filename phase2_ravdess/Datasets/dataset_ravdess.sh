#!/bin/bash
# dataset_ravdess.sh — Download RAVDESS Speech Dataset for Phase 2
#
# Put this file in: Datasets/dataset_ravdess.sh
# Run:  cd Datasets && bash dataset_ravdess.sh
#
# Downloads speech-only portion:
#   1. Audio-only speech (1440 .wav, ~215 MB)
#   2. Video speech per actor (AV+VO .mp4, ~12 GB total)
#   3. Pre-extracted OpenFace facial tracking CSVs (~50 MB)

set -euo pipefail

DEST="ravdess"
mkdir -p "$DEST"

echo "============================================="
echo "  RAVDESS Speech Dataset Download"
echo "============================================="

# ── 1. Audio-only speech (215 MB) ────────────────
echo ""
echo "[1/3] Audio-only speech (~215 MB)..."
if [ ! -f "$DEST/Audio_Speech_Actors_01-24.zip" ]; then
    wget -q --show-progress -O "$DEST/Audio_Speech_Actors_01-24.zip" \
        "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
fi
if [ ! -d "$DEST/audio_speech/Actor_01" ]; then
    mkdir -p "$DEST/audio_speech"
    unzip -q -o "$DEST/Audio_Speech_Actors_01-24.zip" -d "$DEST/audio_speech"
    echo "  Extracted."
else
    echo "  [SKIP] Already extracted."
fi

# ── 2. Video speech (~500 MB x 24 actors) ────────
echo ""
echo "[2/3] Video speech (~12 GB total, 24 actors)..."
mkdir -p "$DEST/video_speech"
for N in $(seq -w 1 24); do
    ZIP="Video_Speech_Actor_${N}.zip"
    DIR="$DEST/video_speech/Actor_${N}"
    if [ -d "$DIR" ] && [ "$(find "$DIR" -name '*.mp4' 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] Actor_${N}"
        continue
    fi
    [ ! -f "$DEST/$ZIP" ] && wget -q --show-progress -O "$DEST/$ZIP" \
        "https://zenodo.org/records/1188976/files/${ZIP}?download=1"
    unzip -q -o "$DEST/$ZIP" -d "$DEST/video_speech"
    echo "  Actor_${N} done."
done

# ── 3. Pre-extracted OpenFace tracking CSVs ──────
echo ""
echo "[3/3] Facial landmark tracking CSVs (~50 MB)..."
if [ ! -f "$DEST/FacialTracking_Actors_01-24.zip" ]; then
    wget -q --show-progress -O "$DEST/FacialTracking_Actors_01-24.zip" \
        "https://zenodo.org/records/3255102/files/FacialTracking_Actors_01-24.zip?download=1"
fi
if [ ! -d "$DEST/facial_tracking/Actor_01" ]; then
    mkdir -p "$DEST/facial_tracking"
    unzip -q -o "$DEST/FacialTracking_Actors_01-24.zip" -d "$DEST/facial_tracking"
    echo "  Extracted."
else
    echo "  [SKIP] Already extracted."
fi

# ── Verify ────────────────────────────────────────
echo ""
echo "============================================="
echo "  Verification"
echo "============================================="
A=$(find "$DEST/audio_speech" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
V=$(find "$DEST/video_speech" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
C=$(find "$DEST/facial_tracking" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
echo "  Audio .wav : $A  (expected 1440)"
echo "  Video .mp4 : $V  (expected 2880)"
echo "  Track .csv : $C  (expected ~2452)"
echo ""
echo "  Filename: MODALITY-CHANNEL-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR"
echo "  Emotion:  01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised"
echo "============================================="