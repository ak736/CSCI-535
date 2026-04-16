#!/usr/bin/env bash
# run_diarization.sh — Step 1 shell wrapper (Readme_Tasks.md interface)
#
# Usage:
#   bash scripts/run_diarization.sh <audio_path> <output_csv>
#
# Example:
#   bash scripts/run_diarization.sh \
#     Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
#     outputs/diarization/ES2002a_diarization.csv

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_diarization.sh <audio_path> <output_csv>" >&2
    exit 1
fi

AUDIO="$1"
OUTPUT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/venv/bin/activate"

python "$REPO_ROOT/charan/charan_diarization.py" \
    --audio "$AUDIO" \
    --output "$OUTPUT"
