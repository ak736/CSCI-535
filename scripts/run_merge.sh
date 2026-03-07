#!/usr/bin/env bash
# run_merge.sh — Step 3 shell wrapper (Readme_Tasks.md interface)
#
# Usage:
#   bash scripts/run_merge.sh <diar_csv> <trans_csv> <out_csv>
#
# Example:
#   bash scripts/run_merge.sh \
#     outputs/diarization/ES2002a_diarization.csv \
#     outputs/transcripts/ES2002a_transcript.csv \
#     outputs/merged/ES2002a_merged.csv

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/run_merge.sh <diar_csv> <trans_csv> <out_csv>" >&2
    exit 1
fi

DIAR="$1"
TRANS="$2"
OUT="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/venv/bin/activate"

python "$REPO_ROOT/aniket/aniket_merge.py" \
    --diar  "$DIAR" \
    --trans "$TRANS" \
    --out   "$OUT"
