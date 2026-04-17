# Phase 2 Guidelines — Aaditya

You own **2 tasks**: video emotion from pre-extracted OpenFace tracking (Step 2b) and final evaluation + plots (Step 6).

You read ONLY from: `phase2_ravdess/metadata/`, `phase2_ravdess/pairs/`, `phase2_ravdess/scores/`, `phase2_ravdess/predictions/`, `Datasets/ravdess/facial_tracking/`.
You write ONLY to: `phase2_ravdess/emotions/video_emotions.csv`, `phase2_ravdess/results/`.

---

## Execution Order

| Day | Step                  | Waits for                          | Script                             |
| --- | --------------------- | ---------------------------------- | ---------------------------------- |
| 2-3 | Step 2b — video emo   | Harish Step 1 (metadata.csv)       | `aaditya_video_emotion.py`         |
| 6-7 | Step 6 — eval + plots | Aniket Step 5 (predictions.csv)    | `aaditya_evaluate_and_plot.py`     |

---

## Step 2b — Video Emotion (pre-extracted OpenFace)

### Script
`phase2_ravdess/scripts/aaditya_video_emotion.py`

### HUGE simplification vs Phase 1
**NO py-feat. NO OpenFace execution. NO `venv_video`.**
The RAVDESS Facial Landmark Tracking dataset ships per-trial CSVs that already contain frame-level AU intensities (OpenFace 2.1.0 output). You just read those CSVs directly with pandas.

### Input
- `phase2_ravdess/metadata/ravdess_metadata.csv` (Harish Step 1)
- `Datasets/ravdess/facial_tracking/Actor_XX/*.csv` — pre-extracted per-clip tracking

### Filename matching (IMPORTANT)
Audio filename uses modality prefix `03`, tracking CSV uses `01`. Everything else is identical.
```
audio    : 03-01-05-02-01-01-08.wav   → clip_id 08_05_02_01_01
tracking : 01-01-05-02-01-01-08.csv   → same clip_id (swap 03 → 01)
```
Helper already present in the stub: `audio_filename_to_tracking_filename()`.

### Columns to read from OpenFace tracking CSV
| Our internal | OpenFace column | Meaning                          |
| ------------ | --------------- | -------------------------------- |
| AU01         | `AU01_r`        | Inner brow raise (intensity 0-5) |
| AU02         | `AU02_r`        | Outer brow raise                 |
| AU04         | `AU04_r`        | Brow lowerer                     |
| AU06         | `AU06_r`        | Cheek raiser                     |
| AU12         | `AU12_r`        | Lip corner puller                |
| AU15         | `AU15_r`        | Lip corner depressor             |
| AU17         | `AU17_r`        | Chin raiser                      |
| AU25         | `AU25_r`        | Lips part                        |

Also use quality columns:
- `confidence` (float, 0-1) — keep frames with `> 0.5`
- `success` (int, 0/1) — keep frames where `== 1`

Note: OpenFace CSVs sometimes emit column names with leading whitespace (`" AU01_r"`). The stub already strips whitespace.

### Processing per clip
1. Load tracking CSV → strip column whitespace.
2. Filter rows where `confidence > 0.5 AND success == 1`.
3. **Mean-pool** each `AU*_r` column across valid frames → a single dict of AU averages per clip.
4. Apply FACS mapping (COPIED from [phase1/scripts/run_video_emotion.py](../phase1/scripts/run_video_emotion.py)):
   ```python
   happy_score   = 0.5*AU06 + 0.5*AU12
   angry_score   = 0.4*AU04 + 0.3*AU17 + 0.3*AU25
   sad_score     = 0.4*AU01 + 0.3*AU04 + 0.3*AU15
   total_act     = AU01+AU02+AU04+AU06+AU12+AU15+AU17+AU25
   neutral_score = max(0, 1 - 0.15 * total_act)
   probs         = softmax([happy, angry, sad, neutral])
   probs         = probs / probs.sum()   # explicit renormalize to 1.0
   ```
5. If tracking CSV is missing OR has zero valid frames → uniform `[0.25, 0.25, 0.25, 0.25]`. Log the clip_id.

### Output
`phase2_ravdess/emotions/video_emotions.csv`:

| Column           | Type   | Example          |
| ---------------- | ------ | ---------------- |
| `clip_id`        | string | `08_05_02_01_01` |
| `actor`          | int    | 8                |
| `emotion_code`   | int    | 5                |
| `emotion_4class` | string | `angry`          |
| `p_happy`        | float  | 0.10             |
| `p_angry`        | float  | 0.45             |
| `p_sad`          | float  | 0.15             |
| `p_neutral`      | float  | 0.30             |

### Run
```bash
source venv/bin/activate
python3 phase2_ravdess/scripts/aaditya_video_emotion.py --help
python3 phase2_ravdess/scripts/aaditya_video_emotion.py \
    --metadata      phase2_ravdess/metadata/ravdess_metadata.csv \
    --tracking_root Datasets/ravdess/facial_tracking \
    --out           phase2_ravdess/emotions/video_emotions.csv
```

### Validation checklist
- [ ] one row per `clip_id` present in metadata
- [ ] `p_happy + p_angry + p_sad + p_neutral == 1.0` per row (tol 1e-4)
- [ ] no NaN
- [ ] log how many clips fell back to uniform (should be small)
- [ ] clip_ids match metadata exactly

---

## Step 6 — Evaluation + Plots

### Script
`phase2_ravdess/scripts/aaditya_evaluate_and_plot.py`

### When to start
Only after Aniket writes both:
- `phase2_ravdess/predictions/predictions.csv`
- `phase2_ravdess/predictions/experiment_log.csv`

### Inputs
- `phase2_ravdess/predictions/predictions.csv`
- `phase2_ravdess/predictions/experiment_log.csv`
- `phase2_ravdess/scores/pair_scores.csv` (for raw-score distributions + emotion-pair heatmap)

### Metrics per ablation config (A, B, C, D)
- ROC-AUC (primary) — mean ± std across 24 folds
- Precision / Recall / F1 at optimal threshold via Youden's J
- Confusion matrix (2x2) for best config at threshold 0.5

Output: `phase2_ravdess/results/evaluation_metrics.csv` — 4 rows (configs A-D) with columns:
```
config, auc_mean, auc_std, f1_mean, precision_mean, recall_mean
```

### Plots (all PNG into `phase2_ravdess/results/plots/`)

| Filename                   | Content                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `roc_curves.png`           | Overlaid ROC curves for configs A-D, AUC in legend              |
| `ablation_delta.png`       | Bar chart: AUC gain A→B→C→D                                     |
| `emotion_pair_heatmap.png` | Heatmap of mean JSD by (audio_emotion, video_emotion) 4×4       |
| `confusion_matrix.png`     | 2×2 congruent/incongruent for best config                       |
| `score_distribution.png`   | Violin/box of JSD for label=0 vs label=1                        |

### Plot style — MATCH Phase 1
Reference: [phase1/scripts/plot_results.py](../phase1/scripts/plot_results.py). Already wired into the stub:
```python
JS_COLOR   = "#2563EB"   # blue
BASE_COLOR = "#DC2626"   # red
DIAG_COLOR = "#9CA3AF"   # gray (chance diagonal)
```
Use sans-serif, 150 dpi figures, savefig at 200 dpi, no top/right spines, bold titles. No icons/emojis.

### Run
```bash
python3 phase2_ravdess/scripts/aaditya_evaluate_and_plot.py \
    --predictions    phase2_ravdess/predictions/predictions.csv \
    --experiment_log phase2_ravdess/predictions/experiment_log.csv \
    --pair_scores    phase2_ravdess/scores/pair_scores.csv \
    --out_metrics    phase2_ravdess/results/evaluation_metrics.csv \
    --plots_dir      phase2_ravdess/results/plots
```

### Validation checklist
- [ ] `evaluation_metrics.csv` has exactly 4 rows (configs A, B, C, D)
- [ ] all 5 PNGs render without error
- [ ] best config AUC is reasonable (> 0.5, probably > 0.7 on controlled data)
- [ ] no NaN in metrics CSV

---

## Before Pushing

```bash
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/aaditya_video_emotion.py').read())"
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/aaditya_evaluate_and_plot.py').read())"

python3 phase2_ravdess/scripts/aaditya_video_emotion.py     --validate_only
python3 phase2_ravdess/scripts/aaditya_evaluate_and_plot.py --validate_only
```

Branch: `phase2/aaditya`.
