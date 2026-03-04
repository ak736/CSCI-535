#!/usr/bin/env bash
# run_whisperx.sh — Step 2 shell wrapper (Readme_Tasks.md interface)
#
# Usage:
#   bash scripts/run_whisperx.sh <audio_path> <output_csv>
#
# Example:
#   bash scripts/run_whisperx.sh \
#     Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
#     outputs/transcripts/ES2002a_transcript.csv

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_whisperx.sh <audio_path> <output_csv>" >&2
    exit 1
fi

AUDIO="$1"
OUTPUT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/venv/bin/activate"

python "$REPO_ROOT/aaditya/aaditya_whisperx.py" \
    --audio "$AUDIO" \
    --output "$OUTPUT"
