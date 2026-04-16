#!/usr/bin/env bash
# aaditya_batch_whisperx.sh — Step 2.3: Batch transcription for all 12 AMI meetings
#
# Usage (run from repo root):
#   bash scripts/aaditya_batch_whisperx.sh
#
# Behavior:
#   - Skips any meeting whose output CSV already exists
#   - Logs progress to outputs/transcripts/batch_run.log (and stdout)
#   - Exits non-zero if any meeting fails

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="$REPO_ROOT/venv/bin/activate"
PYTHON_SCRIPT="$REPO_ROOT/aaditya/aaditya_whisperx.py"
LOG_FILE="$REPO_ROOT/outputs/transcripts/batch_run.log"

MEETINGS=(
    ES2002a ES2002b ES2002c ES2002d
    ES2003a ES2003b ES2003c ES2003d
    ES2004a ES2004b ES2004c ES2004d
)

# Activate venv
if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $VENV" >&2
    exit 1
fi
source "$VENV"

mkdir -p "$REPO_ROOT/outputs/transcripts"

echo "=============================" | tee -a "$LOG_FILE"
echo "Batch start: $(date)" | tee -a "$LOG_FILE"
echo "=============================" | tee -a "$LOG_FILE"

FAILED=()

for MEETING in "${MEETINGS[@]}"; do
    AUDIO="$REPO_ROOT/Datasets/amicorpus/$MEETING/audio/$MEETING.Mix-Headset.wav"
    OUTPUT="$REPO_ROOT/outputs/transcripts/${MEETING}_transcript.csv"

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP]    $MEETING — output already exists: $OUTPUT" | tee -a "$LOG_FILE"
        continue
    fi

    if [ ! -f "$AUDIO" ]; then
        echo "[ERROR]   $MEETING — audio not found: $AUDIO" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING")
        continue
    fi

    echo "[RUNNING] $MEETING — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

    if python "$PYTHON_SCRIPT" \
        --audio "$AUDIO" \
        --output "$OUTPUT" \
        --model small \
        --device mps \
        --batch_size 4 \
        --compute_type int8 2>&1 | tee -a "$LOG_FILE"; then
        echo "[DONE]    $MEETING — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    else
        echo "[FAILED]  $MEETING" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING")
    fi
done

echo "=============================" | tee -a "$LOG_FILE"
echo "Batch end: $(date)" | tee -a "$LOG_FILE"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED meetings: ${FAILED[*]}" | tee -a "$LOG_FILE"
    exit 1
else
    echo "All meetings completed successfully." | tee -a "$LOG_FILE"
fi
