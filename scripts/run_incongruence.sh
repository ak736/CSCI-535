#!/usr/bin/env bash
# run_incongruence.sh — Step 7 shell wrapper (Readme_Tasks.md interface)
#
# Usage:
#   bash scripts/run_incongruence.sh <meeting_id>
#
# Example:
#   bash scripts/run_incongruence.sh ES2002a
#
# Reads:
#   outputs/segments/<meeting>_segments.csv
#   outputs/embeddings/<meeting>_text_emotions.csv
#   outputs/embeddings/<meeting>_audio_emotions.csv
#
# Writes:
#   outputs/incongruence/<meeting>_scores.csv

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_incongruence.sh <meeting_id>" >&2
    echo "Example: bash scripts/run_incongruence.sh ES2002a" >&2
    exit 1
fi

MEETING="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEGMENTS="$REPO_ROOT/outputs/segments/${MEETING}_segments.csv"
TEXT_EMB="$REPO_ROOT/outputs/embeddings/${MEETING}_text_emotions.csv"
AUDIO_EMB="$REPO_ROOT/outputs/embeddings/${MEETING}_audio_emotions.csv"
OUT="$REPO_ROOT/outputs/incongruence/${MEETING}_scores.csv"

# Validate all inputs exist before starting
for f in "$SEGMENTS" "$TEXT_EMB" "$AUDIO_EMB"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: input not found: $f" >&2
        exit 1
    fi
done

source "$REPO_ROOT/venv/bin/activate"

python "$REPO_ROOT/aniket/aniket_incongruence.py" \
    --segments  "$SEGMENTS" \
    --text_emb  "$TEXT_EMB" \
    --audio_emb "$AUDIO_EMB" \
    --out       "$OUT"
