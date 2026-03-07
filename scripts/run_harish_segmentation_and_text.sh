#!/usr/bin/env bash
# run_harish_segmentation_and_text.sh — Batch: Segmentation + Text Emotion Embeddings (Harish)
#
# Phase 1: Loop over all merged CSVs → create segments CSV for each in outputs/segments/
# Phase 2: Loop over all segment CSVs → create text_emotions CSV for each in outputs/embeddings/
#
# Usage (run from repo root):
#   bash scripts/run_harish_segmentation_and_text.sh [out_dir]
#
# Example:
#   bash scripts/run_harish_segmentation_and_text.sh
#   bash scripts/run_harish_segmentation_and_text.sh outputs
#
# Behavior:
#   - Skips a meeting if its segments CSV already exists (Phase 1)
#   - Skips a meeting if its embeddings CSV already exists (Phase 2)
#   - Logs to outputs/segments/batch_run.log (and stdout)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${1:-outputs}"
MERGED_DIR="$REPO_ROOT/$OUT_DIR/merged"
SEGMENTS_DIR="$REPO_ROOT/$OUT_DIR/segments"
EMBEDDINGS_DIR="$REPO_ROOT/$OUT_DIR/embeddings"
LOG_FILE="$REPO_ROOT/$OUT_DIR/segments/batch_run.log"

VENV="$REPO_ROOT/venve/bin/activate"
PYTHON_SCRIPT="$REPO_ROOT/harish/harish_segmentation_and_text.py"

# Same meeting parts as aaditya_batch_whisperx.sh
MEETINGS=(
    ES2002a ES2002b ES2002c ES2002d
    ES2003a ES2003b ES2003c ES2003d
    ES2004a ES2004b ES2004c ES2004d
)

if [ ! -f "$VENV" ]; then
    echo "ERROR: venve not found at $VENV" >&2
    exit 1
fi
source "$VENV"

mkdir -p "$SEGMENTS_DIR" "$EMBEDDINGS_DIR"

echo "=============================" | tee -a "$LOG_FILE"
echo "Harish batch start: $(date)" | tee -a "$LOG_FILE"
echo "  merged_dir:    $MERGED_DIR" | tee -a "$LOG_FILE"
echo "  segments_dir: $SEGMENTS_DIR" | tee -a "$LOG_FILE"
echo "  embeddings:   $EMBEDDINGS_DIR" | tee -a "$LOG_FILE"
echo "=============================" | tee -a "$LOG_FILE"

FAILED=()

# ---------------------------------------------------------------------------
# Phase 1: For each merged CSV → create segments CSV
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "--- Phase 1: Merged → Segments ---" | tee -a "$LOG_FILE"

for MEETING in "${MEETINGS[@]}"; do
    MERGED="$MERGED_DIR/${MEETING}_merged.csv"
    SEGMENTS_OUT="$SEGMENTS_DIR/${MEETING}_segments.csv"

    if [ -f "$SEGMENTS_OUT" ]; then
        echo "[SKIP]    $MEETING — segments already exist: $SEGMENTS_OUT" | tee -a "$LOG_FILE"
        continue
    fi

    if [ ! -f "$MERGED" ]; then
        echo "[ERROR]   $MEETING — merged not found: $MERGED" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING (phase1)")
        continue
    fi

    echo "[RUNNING] $MEETING (segments) — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    if python "$PYTHON_SCRIPT" \
        --merged "$MERGED" \
        --part   "$MEETING" \
        --out_dir "$REPO_ROOT/$OUT_DIR" \
        --segments_only 2>&1 | tee -a "$LOG_FILE"; then
        echo "[DONE]    $MEETING (segments) — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    else
        echo "[FAILED]  $MEETING (segments)" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING (phase1)")
    fi
done

# ---------------------------------------------------------------------------
# Phase 2: For each segments CSV → create embeddings CSV
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "--- Phase 2: Segments → Embeddings ---" | tee -a "$LOG_FILE"

for MEETING in "${MEETINGS[@]}"; do
    SEGMENTS_IN="$SEGMENTS_DIR/${MEETING}_segments.csv"
    EMBEDDINGS_OUT="$EMBEDDINGS_DIR/${MEETING}_text_emotions.csv"

    if [ -f "$EMBEDDINGS_OUT" ]; then
        echo "[SKIP]    $MEETING — embeddings already exist: $EMBEDDINGS_OUT" | tee -a "$LOG_FILE"
        continue
    fi

    if [ ! -f "$SEGMENTS_IN" ]; then
        echo "[ERROR]   $MEETING — segments not found: $SEGMENTS_IN" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING (phase2)")
        continue
    fi

    echo "[RUNNING] $MEETING (embeddings) — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    if python "$PYTHON_SCRIPT" \
        --segments "$SEGMENTS_IN" \
        --part     "$MEETING" \
        --out_dir  "$REPO_ROOT/$OUT_DIR" \
        --embeddings_only 2>&1 | tee -a "$LOG_FILE"; then
        echo "[DONE]    $MEETING (embeddings) — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    else
        echo "[FAILED]  $MEETING (embeddings)" | tee -a "$LOG_FILE"
        FAILED+=("$MEETING (phase2)")
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=============================" | tee -a "$LOG_FILE"
echo "Harish batch end: $(date)" | tee -a "$LOG_FILE"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" | tee -a "$LOG_FILE"
    exit 1
else
    echo "All meetings completed successfully." | tee -a "$LOG_FILE"
fi
