# Phase 2 Guidelines — Aniket

You own **1 task**: train a binary classifier with leave-one-actor-out CV across 4 ablation configurations (Step 5).

You read ONLY from: `phase2_ravdess/scores/`.
You write ONLY to: `phase2_ravdess/predictions/`.

---

## Execution Order

| Day | Step                     | Waits for                           | Script                    |
| --- | ------------------------ | ----------------------------------- | ------------------------- |
| 5-6 | Step 5 — classifier + ablation | Charan Step 4 (pair_scores.csv) | `aniket_classifier.py` |

---

## Step 5 — Classifier + Ablation

### Script
`phase2_ravdess/scripts/aniket_classifier.py`

### Input
`phase2_ravdess/scores/pair_scores.csv` — from Charan Step 4. Contains every pair's features + 4 JSD columns.

### Model
`sklearn.linear_model.LogisticRegression` with:
- `class_weight='balanced'`  (handles any class imbalance)
- `max_iter=2000`
- `random_state=42`

**Standardize features** with `StandardScaler` — fit **only** on train fold, transform test fold (no leakage).

### Cross-validation protocol
`sklearn.model_selection.LeaveOneGroupOut` with `groups = pair_scores["actor"]`.
- 24 actors → 24 folds
- Each fold: train on 23 actors, test on 1
- Prevents speaker-specific leakage entirely

```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    # train on 23 actors, test on 1
    ...
```

### Ablation configurations (exact feature lists)

| Config | Features                                                                      | Count |
| ------ | ----------------------------------------------------------------------------- | ----- |
| A      | `jsd_audio_video`                                                             | 1     |
| B      | `jsd_text_audio`, `jsd_text_video`, `jsd_audio_video`                         | 3     |
| C      | B + `p_audio_happy/angry/sad/neutral` + `p_video_happy/angry/sad/neutral`     | 11    |
| D      | C + `p_text_happy/angry/sad/neutral` + `intensity` + `gender_female` (one-hot)| 17    |

`gender_female` = 1 if gender == `"female"` else 0 (one-hot encoded in `prepare_features()`; stub already wires this).

### Outputs

**Output 1: `phase2_ravdess/predictions/predictions.csv`** — one row per (pair × config):
| Column       | Type   | Notes                                                |
| ------------ | ------ | ---------------------------------------------------- |
| `pair_id`    | string | From pair_scores                                     |
| `label`      | int    | Ground truth (0 congruent, 1 incongruent)            |
| `pred_prob`  | float  | `clf.predict_proba(X_test)[:, 1]` — must be in [0,1] |
| `pred_label` | int    | `(pred_prob >= 0.5).astype(int)`                     |
| `config`     | string | A / B / C / D                                        |
| `fold_actor` | int    | The actor held out for this row                      |

Row count = `len(pair_scores) * 4` (each pair appears in each config, once per fold it belongs to).

**Output 2: `phase2_ravdess/predictions/experiment_log.csv`** — exactly 4 rows:
| Column           | Notes                                          |
| ---------------- | ---------------------------------------------- |
| `config`         | A / B / C / D                                  |
| `auc_mean`       | Mean ROC-AUC across the 24 folds               |
| `auc_std`        | Std of AUC across folds                        |
| `f1_mean`        | Mean F1 across folds at threshold 0.5          |
| `precision_mean` | Mean precision                                 |
| `recall_mean`    | Mean recall                                    |

Note: if a fold has only one class in its test set, skip it when computing AUC (document in code).

### Run
```bash
source venv/bin/activate
python3 phase2_ravdess/scripts/aniket_classifier.py --help
python3 phase2_ravdess/scripts/aniket_classifier.py \
    --scores          phase2_ravdess/scores/pair_scores.csv \
    --out_predictions phase2_ravdess/predictions/predictions.csv \
    --out_experiment  phase2_ravdess/predictions/experiment_log.csv
```

### Validation checklist
- [ ] `predictions.csv` contains all 4 configs × 24 folds
- [ ] `pred_prob` values in `[0.0, 1.0]`
- [ ] `experiment_log.csv` has exactly 4 rows (configs A–D)
- [ ] all AUC values > 0.5 (sanity check — above chance)
- [ ] no NaN in either CSV

---

## Before Pushing

```bash
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/aniket_classifier.py').read())"
python3 phase2_ravdess/scripts/aniket_classifier.py --validate_only
```

Branch: `phase2/aniket`.
