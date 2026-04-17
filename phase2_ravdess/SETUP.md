# Phase 2 Setup Guide

Read this first. Then your personal `GUIDELINES_*.md`. Then the master [README_Phase2_RAVDESS.md](README_Phase2_RAVDESS.md).

---

## Reading order

1. **SETUP.md** (this file) — environment, dataset, folder layout, branches
2. **[README_Phase2_RAVDESS.md](README_Phase2_RAVDESS.md)** — full plan, source of truth
3. **Your personal guideline**:
   - Harish  → [GUIDELINES_HARISH.md](GUIDELINES_HARISH.md)
   - Charan  → [GUIDELINES_CHARAN.md](GUIDELINES_CHARAN.md)
   - Aaditya → [GUIDELINES_AADITYA.md](GUIDELINES_AADITYA.md)
   - Aniket  → [GUIDELINES_ANIKET.md](GUIDELINES_ANIKET.md)
4. **Your scripts** in `phase2_ravdess/scripts/`

---

## 1. Folder structure

The scaffold is already created. Layout:

```
phase2_ravdess/
├── README_Phase2_RAVDESS.md         (master plan — frozen)
├── SETUP.md                          (this file)
├── GUIDELINES_HARISH.md
├── GUIDELINES_CHARAN.md
├── GUIDELINES_AADITYA.md
├── GUIDELINES_ANIKET.md
├── scripts/                          (all 8 Python scripts)
├── metadata/                         (Harish writes)
├── emotions/                         (Harish, Charan, Aaditya each write one CSV)
├── pairs/                            (Harish writes)
├── scores/                           (Charan writes)
├── predictions/                      (Aniket writes)
└── results/
    └── plots/                        (Aaditya writes)
```

Each person writes ONLY to their designated folders (spelled out in each `GUIDELINES_*.md`). Read from anyone's; write to yours.

---

## 2. Download the dataset

```bash
cd "CSCI 535 MPL/Implementation/Datasets"
bash dataset_ravdess.sh
```

This downloads:

| What                | Size    | Files       | Path                                         |
| ------------------- | ------- | ----------- | -------------------------------------------- |
| Audio-only speech   | ~215 MB | 1440 .wav   | `Datasets/ravdess/audio_speech/Actor_XX/`    |
| Video speech        | ~12 GB  | 2880 .mp4   | `Datasets/ravdess/video_speech/Actor_XX/`    |
| OpenFace tracking   | ~50 MB  | ~2452 .csv  | `Datasets/ravdess/facial_tracking/Actor_XX/` |

### Verify downloads
```bash
find Datasets/ravdess/audio_speech    -name "*.wav" | wc -l   # must be 1440
find Datasets/ravdess/facial_tracking -name "*.csv" | wc -l   # must be ~2452
```

**Disk-constrained?** Audio (~215 MB) + tracking (~50 MB) is the minimum. The .mp4s are optional — we use the pre-extracted tracking CSVs, not raw video.

---

## 3. Activate the Python environment

Phase 2 uses the **same venv as Phase 1**. No new packages needed.

```bash
cd "CSCI 535 MPL/Implementation"
source venv/bin/activate
```

The venv already has: torch, transformers, librosa, pandas, numpy, scikit-learn, matplotlib, soundfile.

Aaditya does NOT need `venv_video` or py-feat — the tracking CSVs are plain pandas-readable.

---

## 4. Verify script scaffolding

Every stub script supports `--help` and `--validate_only`. Sanity check:
```bash
for s in phase2_ravdess/scripts/*.py; do
    python3 -c "import ast; ast.parse(open('$s').read())" && echo "OK: $s"
done

python3 phase2_ravdess/scripts/harish_parse_metadata.py --help
```

---

## 5. Git branch strategy

```
master                 ← Phase 1 code in phase1/, frozen — do not push here directly
phase2/harish          ← metadata + text emotion + pair construction
phase2/charan          ← audio emotion + JSD scoring
phase2/aaditya         ← video emotion + evaluation / plots
phase2/aniket          ← classifier + ablation
```

Create your branch:
```bash
git checkout -b phase2/<your-name>
```

Merge to `master` only after everything is validated.
No merge conflicts expected — each person owns a disjoint set of files + output folders.

---

## 6. The 7 rules (from the master README)

1. **`clip_id` is the universal join key.** Format: `{actor:02d}_{emotion_code:02d}_{intensity:02d}_{statement:02d}_{repetition:02d}`.
2. **4-class label space only.** `E = {happy, angry, sad, neutral}` — always these exact 4 columns `p_happy, p_angry, p_sad, p_neutral`, summing to 1.0.
3. **No NaN, no duplicate clip_ids.** Validate before handing off.
4. **Carry ground truth.** Every CSV must include `emotion_code` (int 1-8) and `emotion_4class` (string) so downstream can verify without re-joining metadata.
5. **Do not touch Phase 1.** Everything in `phase1/` is frozen.
6. **Write only to your own folders.** Read from anyone; write to the paths listed in your guideline.
7. **Use `np.random.seed(42)`** at the top of every script (already wired in the stubs).

---

## 7. Quick start per person

| Person  | First command                                                                   |
| ------- | ------------------------------------------------------------------------------- |
| Harish  | `python3 phase2_ravdess/scripts/harish_parse_metadata.py --help`                |
| Charan  | `python3 phase2_ravdess/scripts/charan_audio_emotion.py --help`                 |
| Aaditya | `python3 phase2_ravdess/scripts/aaditya_video_emotion.py --help`                |
| Aniket  | `python3 phase2_ravdess/scripts/aniket_classifier.py --help`                    |

Open your `GUIDELINES_*.md` in one pane, the matching script in another, and start filling in the `TODO` blocks.
