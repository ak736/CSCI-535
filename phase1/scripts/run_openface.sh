#!/bin/bash
# run_openface.sh — Wrapper for video feature extraction
#
# Since OpenFace is not installed, this script delegates to
# extract_video_features.py which uses py-feat (Python-based
# AU extraction) as an equivalent replacement.
#
# py-feat extracts the same features as OpenFace:
#   - Action Units (AU01-AU25 intensities)
#   - Head pose (pitch, yaw, roll)
#   - Gaze direction (x, y)
#
# Usage:
#   bash scripts/run_openface.sh ES2002a
#   bash scripts/run_openface.sh --all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the Python 3.10 venv with py-feat
VENV_PATH="$PROJECT_DIR/venv_video"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: venv_video not found at $VENV_PATH"
    echo "Create it with: python3.10 -m venv venv_video && pip install py-feat opencv-python pandas numpy"
    exit 1
fi

if [ "${1:-}" = "--all" ]; then
    for meeting in ES2002a ES2002b ES2002c ES2002d ES2003a ES2003b ES2003c ES2003d ES2004a ES2004b ES2004c ES2004d; do
        echo "=== Processing $meeting ==="
        python "$SCRIPT_DIR/extract_video_features.py" --meeting "$meeting"
    done
elif [ -n "${1:-}" ]; then
    python "$SCRIPT_DIR/extract_video_features.py" --meeting "$1"
else
    echo "Usage: bash scripts/run_openface.sh MEETING_ID"
    echo "       bash scripts/run_openface.sh --all"
    exit 1
fi
