#!/usr/bin/env bash
# run_audio_emotion.sh — Step 6 shell wrapper (Readme_Tasks.md interface)
#
# Usage:
#   bash scripts/run_audio_emotion.sh <meeting_id>
#
# Example:
#   bash scripts/run_audio_emotion.sh ES2002a
#
# Runs audio emotion prediction for a single meeting using the individual
# headset WAV files (speaker-aware) and writes:
#   outputs/embeddings/<meeting>_audio_emotions.csv

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_audio_emotion.sh <meeting_id>" >&2
    echo "Example: bash scripts/run_audio_emotion.sh ES2002a" >&2
    exit 1
fi

MEETING="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEGMENTS="$REPO_ROOT/outputs/segments/${MEETING}_segments.csv"
DIAR="$REPO_ROOT/outputs/diarization/${MEETING}_diarization.csv"
AUDIO_DIR="$REPO_ROOT/Datasets/amicorpus/${MEETING}/audio"
OUT="$REPO_ROOT/outputs/embeddings/${MEETING}_audio_emotions.csv"

# Validate inputs before starting
if [ ! -f "$SEGMENTS" ]; then
    echo "ERROR: segments CSV not found: $SEGMENTS" >&2
    exit 1
fi
if [ ! -d "$AUDIO_DIR" ]; then
    echo "ERROR: audio directory not found: $AUDIO_DIR" >&2
    exit 1
fi

source "$REPO_ROOT/venv/bin/activate"

python "$REPO_ROOT/aniket/aniket_audio_emotion.py" \
    --segments  "$SEGMENTS" \
    --diar      "$DIAR" \
    --audio_dir "$AUDIO_DIR" \
    --meeting   "$MEETING" \
    --out       "$OUT"
