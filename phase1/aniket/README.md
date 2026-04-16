# Step 3: Merge Diarization + Transcript — Aniket

Pipeline position: **Step 2 (Transcription) → Step 3 (This) → Step 4 (Segmentation)**

---

## What This Does

Takes the two separately-produced CSVs — diarization (who spoke, when) and transcription (what was said, when) — and joins them into a single merged CSV by assigning a `speaker_id` to each transcript segment.

The two CSVs come from different models and have **different time boundaries**, so rows cannot simply be zipped together. Instead, for each transcript segment, the script finds the diarization speaker with the greatest time overlap.

**Example:**

```
Diarization:  [4.77 ——— 7.83]  SPEAKER_02
              [7.83 ————————— 9.46]  SPEAKER_03

Transcript:   [4.77 ————————— 7.98]  "My gosh, you've already produced a PowerPoint presentation."

Overlap with SPEAKER_02 = 7.83 - 4.77 = 3.06s  ✓ winner
Overlap with SPEAKER_03 = 7.98 - 7.83 = 0.15s
→ Assigned: SPEAKER_02
```

---

## Files

| File | Description |
|------|-------------|
| `aniket/aniket_merge.py` | Main Python script — overlap matching logic |
| `scripts/run_merge.sh` | Shell wrapper (team interface) |

---

## Output Format

**Location:** `outputs/merged/ESXXXX_merged.csv`

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | int | 0-indexed, re-indexed from 0 |
| `start` | float | Segment start time in seconds (from transcript) |
| `end` | float | Segment end time in seconds (from transcript) |
| `speaker_id` | str | e.g. `SPEAKER_00` — assigned via max overlap with diarization |
| `transcript` | str | Transcribed text (from WhisperX) |
| `asr_confidence` | float | 0.0–1.0 ASR confidence score (from WhisperX) |

**Example row:**
```
0,4.77,7.98,SPEAKER_02,"My gosh, you've already produced a PowerPoint presentation.",0.5756
```

**Already produced:** All 12 meetings in `outputs/merged/`

---

## How to Run

### Standard interface (use this)
```bash
bash scripts/run_merge.sh \
  outputs/diarization/ES2002a_diarization.csv \
  outputs/transcripts/ES2002a_transcript.csv \
  outputs/merged/ES2002a_merged.csv
```

### Direct Python
```bash
python3 aniket/aniket_merge.py \
  --diar  outputs/diarization/ES2002a_diarization.csv \
  --trans outputs/transcripts/ES2002a_transcript.csv \
  --out   outputs/merged/ES2002a_merged.csv
```

### Batch all 12 meetings
```bash
for meeting in ES2002a ES2002b ES2002c ES2002d ES2003a ES2003b ES2003c ES2003d ES2004a ES2004b ES2004c ES2004d; do
  python3 aniket/aniket_merge.py \
    --diar  outputs/diarization/${meeting}_diarization.csv \
    --trans outputs/transcripts/${meeting}_transcript.csv \
    --out   outputs/merged/${meeting}_merged.csv
done
```

---

## Merge Logic

1. For each transcript segment `[t_start, t_end]`:
   - Compute time overlap with every diarization segment
   - Assign `speaker_id` of the diarization segment with the **maximum overlap**
2. **Fallback** (if no diarization segment overlaps — gap/silence):
   - Find the nearest diarization segment by midpoint distance
   - This ensures **no row is ever left without a speaker**

---

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--diar` | Yes | Path to diarization CSV |
| `--trans` | Yes | Path to transcript CSV |
| `--out` | Yes | Path to output merged CSV |

---

## Meetings Processed

| Meeting | Diarization Segments | Transcript Segments | Merged Rows |
|---------|---------------------|---------------------|-------------|
| ES2002a | 246 | 201 | 201 |
| ES2002b | 289 | 436 | 436 |
| ES2002c | 362 | 503 | 503 |
| ES2002d | 571 | 660 | 660 |
| ES2003a | 106 | 160 | 160 |
| ES2003b | 183 | 406 | 406 |
| ES2003c | 232 | 377 | 377 |
| ES2003d | 435 | 443 | 443 |
| ES2004a | 164 | 230 | 230 |
| ES2004b | 329 | 427 | 427 |
| ES2004c | 314 | 461 | 461 |
| ES2004d | 461 | 513 | 513 |

---

## Notes for Downstream Steps

- **Harish (Step 4 — Segmentation):** Use `start`/`end` and `speaker_id` to create 3.0s windows with 1.5s hop. Filter out multi-speaker windows and segments with < 3 words.
- **Text Emotion (Step 5):** Use the `transcript` column as input. `asr_confidence` flows through for confidence weighting.
- **Audio Emotion (Step 6):** Use `start`/`end` timestamps to slice the original `.wav` file per segment.
- **Incongruence Scorer (Step 7):** `asr_confidence` is used in the confidence weighting formula — low-confidence transcript rows will be downweighted.
