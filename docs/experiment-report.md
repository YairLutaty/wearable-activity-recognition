# Wearable Activity Recognition — Experiment Report

Status as of 2026-08-13. The original 18-class model-selection cycle and the
separate business-driven grouped-target extension are complete. Each was
evaluated once on the reserved subjects, and both test evaluations are closed
to further model or policy changes. This document is a complete handoff
description of the project: the data, the pipeline, the constraints, what has
been established, and what is still open. It is written to be read cold by
someone who has not seen the code.

---

## 1. Goal

Classify human activity from smartphone and smartwatch inertial sensors.

- **18 activity classes** (WISDM-style label codes; note there is no `N`):

  | Code | Activity | Code | Activity | Code | Activity |
  |---|---|---|---|---|---|
  | A | Walking | G | Brushing Teeth | M | Kicking Soccer Ball |
  | B | Jogging | H | Eating Soup | O | Playing Catch |
  | C | Stairs | I | Eating Chips | P | Dribbling Basketball |
  | D | Sitting | J | Eating Pasta | Q | Writing |
  | E | Standing | K | Drinking from Cup | R | Clapping |
  | F | Typing | L | Eating Sandwich | S | Folding Clothes |

- **51 subjects**, IDs 1600–1650.
- **4 sensor streams** per subject/activity: phone-accel, phone-gyro,
  watch-accel, watch-gyro.
- Each recording is nominally ~180 s per subject/activity/stream.
- Evaluation must be **subject-independent** — no subject may appear in more
  than one partition.

---

## 2. Repository layout

```
activity_project/
├── raw/                              # 204 source .txt files (51 per stream), 844 MB — gitignored
│   ├── phone/{accel,gyro}/data_<id>_<sensor>_<device>.txt
│   └── watch/{accel,gyro}/data_<id>_<sensor>_<device>.txt
│
├── eda/
│   ├── eda.ipynb                     # 167 cells, 11 sections, fully executed
│
├── data_clean.parquet                # 256 MB, EDA-only cleaned long table — gitignored
│
├── baseline/
│   ├── shared_train_test_samples.py  # frozen windowing + split script
│   ├── baseline.ipynb                # 79 cells, classical baselines and diagnostics
│   └── artifacts/
│       ├── samples.parquet           # 330 MB frozen windowed table — gitignored
│       ├── split_meta.json           # protocol constants + subject lists
│       └── development_folds.json     # fixed 5-fold development-subject mapping
│
├── preprocessing/
│   ├── preprocessing.ipynb           # shared sequence preprocessing, complete
│
├── modeling/
│   ├── 01_rf_parity.ipynb            # resampled-vs-raw RF parity experiment, complete
│   ├── 02_engineered_rf.ipynb         # controlled engineered-feature RF experiment, complete
│   ├── 03_cnn.ipynb                   # compact sequence-model experiment, complete
│   ├── 04_tabular_models.ipynb        # 222-feature learner comparison, complete
│   ├── 05_feature_mlp.ipynb            # MLP on 222 engineered features, complete
│   ├── 06_final_evaluation.ipynb       # frozen model and reserved-test evaluation, complete
│   ├── engineered_features.py          # shared deterministic 222-feature extractor
│   ├── artifacts/                     # generated modeling tables — large files gitignored
│   │   ├── engineered_222_meta.json   # tracked 222-feature artifact contract
│   │   ├── final_model_spec.json       # frozen HGB specification before final test
│   │   └── final_test_result.json      # tracked one-time final evaluation record
│
├── business_case/                    # grouped-target business extension
│   ├── 01_grouped_target.ipynb       # grouped-label model selection, complete
│   ├── 02_business_policy.ipynb      # confidence/event policy selection, complete
│   ├── 03_test_evaluation.ipynb      # frozen grouped pipeline's final test, complete
│   ├── BUSINESS_GOAL.md              # product scenario, actions, metrics, limitations
│   └── artifacts/
│       ├── activity_group_mapping.json # tracked 18-to-3 target contract
│       ├── grouped_target_result.json  # tracked development result and selection
│       ├── business_policy.json        # frozen policy before grouped-target test
│       ├── grouped_test_result.json     # tracked final grouped-test record
│       ├── direct_oof_predictions.parquet # local OOF probabilities — gitignored
│       ├── grouped_final_model.joblib   # fitted final grouped model — gitignored
│       ├── grouped_test_predictions.parquet # window predictions — gitignored
│       └── grouped_test_recordings.parquet # recording summaries — gitignored
│
└── modified/                         # staging area for the datasets used downstream
    ├── build_windowed_splits.py      # freezes train/test out of samples.parquet
    ├── train_samples.parquet         # 297 MB total with test — gitignored
    ├── test_samples.parquet
    ├── windowed_split_meta.json
    └── watch_pair_20hz/
        ├── train_sequences.npz       # paired watch arrays + masks — gitignored
        ├── test_sequences.npz        # paired watch arrays + masks — gitignored
        ├── train_metadata.parquet    # array-row provenance — gitignored
        ├── test_metadata.parquet     # array-row provenance — gitignored
        └── preprocessing_meta.json   # tracked preprocessing contract
```

`modified/` is the designated home for the train/test sets actually used for
modelling, and for preprocessed variants of them as they are produced.

---

## 3. Frozen evaluation protocol

The train/test split and windowing protocol were frozen before model
development. They must be reproduced exactly so results remain comparable
across every experiment, regardless of what later EDA might otherwise suggest.

`baseline/shared_train_test_samples.py` treats these values as a fixed protocol.

**Frozen protocol:**

| Parameter | Value |
|---|---|
| Window length | 5.0 s, non-overlapping |
| Window boundaries | cut within each contiguous activity bout, relative to bout start |
| Incomplete final window | dropped |
| Minimum samples/window | 50 |
| Duplicate handling | exact duplicate rows dropped per file before windowing |
| Test subjects (10, fixed) | 1606, 1607, 1618, 1626, 1629, 1639, 1640, 1641, 1643, 1646 |
| Train/dev subjects | the remaining 41 |
| `window_id` format | `{user_id}_{device}_{sensor}_{activity}_{bout_window:04d}` |

A window is **sensor-specific**: phone-accel, phone-gyro, watch-accel and
watch-gyro segments have different `window_id`s and have not been combined into
a multimodal example.

**What is locked vs. what is free:**

- **Locked:** which subjects are test; 5 s non-overlapping windows; ≥50 samples;
  `window_id` format. `samples.parquet` is the shared artifact.
- **Free:** the internal cross-validation folds over the 41 dev subjects (the
  team fixed only the *test* users); feature engineering; which streams to use;
  fusion strategy; model class; any preprocessing applied **downstream** of the
  frozen table.
- **Consequence:** window *length* and non-overlap cannot change without
  breaking comparability. Resampling *within* a 5 s window changes no canonical
  boundary or window count. Any overlapping-window augmentation would be a
  separate training experiment and should require explicit team agreement;
  evaluation must remain on the canonical windows.

---

## 4. Data pipeline, stage by stage

### Stage 1 — Raw ingest and cleaning (`eda/eda.ipynb`, section 1)

- Raw rows: **15,630,426**
- Exact duplicate rows removed: **263,696** (1.687%) — all belonging to
  **subject 1629**, whose measurements appear twice across every activity and
  stream. Subject 1629 is one of the ten test subjects, so this matters.
- Clean rows: **15,366,730** → cached as `data_clean.parquet`
- No missing values anywhere. No repeated timestamps.
- Rows sorted chronologically within each subject/activity/device/sensor.

### Stage 2 — Windowing and split (`baseline/shared_train_test_samples.py`)

Reads `raw/` directly (not `data_clean.parquet`) and applies the frozen
protocol. Its per-file `drop_duplicates()` is equivalent in effect to the EDA's
global dedup, so the two paths agree on what a valid row is. The script does not
sort by timestamp — it relies on file order — but this was verified downstream
(zero non-increasing timestamps within windows).

Output `baseline/artifacts/samples.parquet`, long format, one row per raw sensor
sample:

`split, user_id, activity, device, sensor, window_id, window_idx, timestamp, x, y, z`

| | Rows | Windows | Subjects |
|---|---|---|---|
| Train | 11,921,624 | 102,729 | 41 |
| Test | 3,063,648 | 24,772 | 10 |
| **Total** | **14,985,272** | **127,501** | **51** |

Built-in integrity checks: every window has ≥50 samples, `window_idx` is a
complete 0-based range, no subject spans both splits, test set equals the fixed
list.

### Stage 3 — Freeze (`modified/build_windowed_splits.py`)

Splits `samples.parquet` into `train_samples.parquet` / `test_samples.parquet`
(zstd), asserts exact row counts and that the subject lists match
`split_meta.json`, refuses to overwrite existing outputs, and writes
`windowed_split_meta.json`.

### Stage 4 — Shared paired-watch preprocessing (`preprocessing/preprocessing.ipynb`)

Reads only the frozen train/test Parquet files, applies the fixed temporal
quality rules, pairs watch accelerometer and gyroscope windows, and resamples
each sensor independently by timestamp to a 20 Hz grid of 100 positions from
0.00 through 4.95 seconds.

- Six development sensor windows fail quality checks: three accelerometer and
  three gyroscope windows at the same three recording positions. No test window
  fails.
- After quality filtering, 120 development and 79 test accelerometer windows
  lack gyroscope partners and are not synthesized.
- Output population: **25,585 development pairs** and **6,161 test pairs**,
  covering all expected subjects and all 18 activities.
- Each split stores separate unnormalized `float32` accelerometer and gyroscope
  arrays with shape `(pairs, 100, 3)`, sensor-specific edge-coverage masks, and
  the target-time grid.
- Linear timestamp interpolation is used within the observed interval;
  out-of-range edge positions receive the nearest observed value. Each sensor
  uses its own nominal timeline, avoiding unjustified sample-level alignment
  between accelerometer and gyroscope.
- Approximately 0.97% of target positions are edge padded; the 99th percentile
  is one padded position per window. Development interpolation brackets above
  0.10 seconds occur in fewer than 1% of windows and remain within the fixed
  0.5-second quality limit.
- A 36-column row-metadata table preserves array index, label, subject, fold,
  source IDs, timing, padding, and interpolation diagnostics. The JSON manifest
  records the full preprocessing contract and output counts.

The resulting files are under `modified/watch_pair_20hz/`. The 20 Hz rate is a
documented development choice, not a change to the frozen canonical window
protocol. Its parity with the classical benchmark was checked using only the
fixed development folds in the next stage.

### Stage 5 — Resampled Random Forest parity (`modeling/01_rf_parity.ipynb`)

Recreates the original 56 fused watch features from the 20 Hz arrays: mean,
sample standard deviation, minimum, maximum, median, IQR, and RMS for each of
`x`, `y`, `z`, and magnitude, separately for accelerometer and gyroscope. It
uses the same paired examples, five subject folds, and 300-tree Random Forest.

| Representation | Mean accuracy | Mean balanced accuracy | Mean macro-F1 | Fold std |
|---|---:|---:|---:|---:|
| Original irregular samples | 0.6920 | 0.6929 | 0.6907 | 0.0480 |
| 20 Hz resampled sequences | 0.6899 | 0.6909 | **0.6882** | 0.0509 |

The macro-F1 difference is only **-0.0025**, comfortably passing the
predeclared parity threshold of 0.67. Fold-level differences are all within
0.01, so the mean does not conceal a failed subject fold. The 20 Hz artifact is
therefore accepted as the shared input for subsequent development experiments.
This establishes parity for the original whole-window statistics; it does not
prove that 20 Hz retains every high-frequency temporal pattern or that it is the
optimal sampling rate for a sequence model.

### Stage 6 — Engineered-feature Random Forest (`modeling/02_engineered_rf.ipynb`)

Compares five predeclared feature sets using the same paired examples, folds,
metrics, and 300-tree Random Forest. In addition to the 56 parity features, the
experiment defines distribution/correlation, temporal-rate, and Hann-window
periodogram feature groups.

| Feature set | Features | Mean macro-F1 | Fold std | Gain vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 56 | 0.6882 | 0.0509 | — |
| + distribution/correlation | 110 | 0.7044 | 0.0501 | +0.0162 |
| + temporal dynamics | 96 | 0.7070 | 0.0485 | +0.0188 |
| + frequency structure | 128 | 0.7217 | 0.0473 | +0.0335 |
| **All engineered groups** | **222** | **0.7257** | **0.0449** | **+0.0374** |

Every engineered candidate improves all five subject folds. Frequency features
provide most of the gain, while the full feature set is strongest overall and
improves 17 of 18 activities and 38 of 41 held-out development subjects versus
the baseline. Writing declines slightly (F1 -0.007); Eating Sandwich remains
the weakest class despite improving from 0.252 to 0.287. The full set is the
new classical development benchmark. The reserved test data remain untouched.

### Stage 7 — Compact 1D CNN (`modeling/03_cnn.ipynb`)

Compares a watch-accelerometer CNN with a dual-branch watch accelerometer–
gyroscope CNN. Each outer subject fold uses a disjoint inner subject fold for
epoch selection; normalization is fitted only on the applicable training
subjects. The dual model improves the seed-42 mean macro-F1 from 0.6536 to
0.6760 and improves 4 of 5 folds, passing the predeclared modality-fusion gate.

The selected dual model is then repeated across seeds 42, 43, and 44. Its
seed-level mean macro-F1 values are 0.6760, 0.6773, and 0.6753, for an overall
mean of **0.6762** and a standard deviation of **0.0010 across seed means**.
Despite this strong stochastic stability, its mean performance is 0.0495 below
the engineered Random Forest and loses to it on all five folds. The CNN fails
the predeclared improvement gate, so the 222-feature Random Forest remains the
strongest development model. The reserved test data remain untouched.

### Stage 8 — Tabular learners on engineered features (`modeling/04_tabular_models.ipynb`)

Keeps the 222-feature representation fixed and compares two new learning
algorithms against the engineered Random Forest. Fold-specific standardization
is used for logistic regression; histogram gradient boosting operates on the
unscaled features with one fixed configuration and row-wise automatic early
stopping disabled.

| Learner | Mean macro-F1 | Fold std | Gain vs RF | Folds above RF |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7126 | 0.0466 | -0.0131 | 1/5 |
| Engineered Random Forest | 0.7257 | 0.0449 | — | — |
| **Histogram gradient boosting** | **0.7446** | **0.0432** | **+0.0189** | **5/5** |

The linear result shows that much of the engineered representation is linearly
separable, while nonlinear learning remains beneficial. Histogram gradient
boosting passes the predeclared replacement gate and becomes the strongest
development model. The reserved test data remain untouched.

### Stage 9 — Feature MLP (`modeling/05_feature_mlp.ipynb`)

Evaluates one predeclared 92,306-parameter MLP on the same 222 features. Each
outer fold uses subject-disjoint inner epoch selection and fold-specific
standardization. At seed 42 it reaches mean macro-F1 **0.7341 ± 0.0528**, which
is 0.0105 below histogram gradient boosting and wins only fold 3. It also falls
0.0005 below the predeclared threshold for repeating additional seeds, so the
experiment stops as planned.

The optional stronger temporal experiment is deferred. Histogram gradient
boosting is frozen as the final model for this selection cycle. The shared
feature module reproduces every value in the saved 25,585 × 222 development
feature table exactly; `modeling/artifacts/final_model_spec.json` records the
configuration before reserved-test access.

### Stage 10 — Final reserved-test evaluation (`modeling/06_final_evaluation.ipynb`)

Fits the frozen histogram-gradient-boosting model on all 25,585 development
examples from 41 subjects and evaluates it once on 6,161 examples from the ten
reserved subjects. The final result is accuracy **0.6780**, balanced accuracy
**0.6773**, and macro-F1 **0.6851**. No post-test model or pipeline changes are
made.

Performance is uneven across both activities and subjects. Jogging reaches F1
0.973, while Eating Sandwich reaches 0.175 and participates in a broad eating-
activity confusion cluster. Subject-level macro-F1 ranges from 0.2059 to
0.8706 with a median of 0.7658; subjects 1639 and 1640 are the main difficult
outliers. `modeling/artifacts/final_test_result.json` contains the exact final
metrics and checksums for the gitignored model and prediction artifacts.

---

## 5. EDA findings (`eda/eda.ipynb`)

Eleven sections: load/clean → coverage → timing → synchronization →
distributions → raw signals → magnitude → window-level → frequency-domain →
subject variability → conclusions.

### Coverage
- All 51 subjects and all 18 activities present.
- **4 subject/activity pairs completely absent:** 1616/Jogging,
  1618/Playing Catch, 1642/Stairs, 1642/Typing.
- **35 of 3,672 possible device–sensor streams missing.** Most subject/activity
  pairs nevertheless have all four streams.
- Decision taken: keep these subjects, leave gaps missing, never synthesize.

### Timing
- Median stream span **179.85 s**, tightly concentrated near 180 s; at least one
  stream reaches **351.32 s**.
- **No single uniform sampling cadence.** Medians near 20 Hz for all four
  device–sensor combinations, but stream-level groups also appear near **25 Hz
  and 50 Hz**, with burst-like local cadences near **100 Hz** that sustain only
  ~33–50 Hz effectively. The subject heatmap shows these differences are
  **systematic by subject and device**, suggesting multiple recording
  configurations or collection cohorts.
- Maximum observed intra-stream gap: **56.66 s** (subject 1649, Kicking Soccer
  Ball).
- **Implication:** a fixed number of rows does not represent a fixed duration.
  Use timestamps, not row positions. Long gaps are sequence boundaries.

### Synchronization
- **Within-device accel↔gyro is aligned:** median phone start difference ~0.15 s
  (end effectively 0), median watch start/end differences ~0.05 s. One phone
  outlier near 861 s needs separate investigation.
- **Phone and watch clocks are not comparable.** Raw phone–watch offsets range
  from ~274 s to >2,000,000 s, median **~161,781 s**, for recordings that last
  ~180 s. These are clock-origin offsets, not misalignment. Largest offsets
  concentrate in subject 1650.
- **Implication:** within-device fusion is straightforward; phone↔watch fusion
  requires a separate, validated alignment procedure.

### Signals
- Locomotion activities produce strong magnitude and clearly periodic patterns.
- The **watch is especially informative for hand-oriented activities**; phone and
  watch carry complementary information.
- Many static and fine-motor activities overlap heavily in simple magnitude
  summaries.
- Dominant-frequency estimates are useful for rhythmic motion but unreliable for
  weak, static or irregular signals — frequency features should use peak
  strength/prominence and band energies rather than forcing one dominant
  frequency per window.

### Subject variability
- The same labeled activity produces meaningfully different motion across
  subjects, and the differences are **activity-specific** rather than a single
  per-subject "moves more / moves less" tendency.
- Largest coefficient of variation: Eating Sandwich ~0.83, Drinking from Cup
  ~0.74, Standing ~0.65, Sitting ~0.64 — mostly low-motion activities where
  subject differences are large *relative* to a small mean.
- Largest *absolute* between-subject spread: Jogging (~1.39), but its CV is only
  ~0.18 because its mean is high.
- **Implication:** split by subject, report per-activity metrics and the
  distribution across held-out subjects, and never drop a subject for being
  different from the population average.

---

## 6. Baseline notebook (`baseline/baseline.ipynb`)

Structure: load → understand train structure → pre-model window checks →
reproducible evaluation protocol → phone-accelerometer baseline → four-stream
diagnostic → controlled within-watch feature fusion.

### Window quality checks (training split)

All windows pass metadata and index integrity; train/test subjects are disjoint.

| Metric | mean | min | 1% | 50% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|
| samples/window | 116.05 | 57 | 99 | 100 | 248 | 249 | 504 |
| observed span (s) | 4.954 | 2.797 | 4.934 | 4.946 | 4.992 | 4.997 | 5.000 |
| max internal gap (s) | 0.047 | 0.010 | 0.020 | 0.050 | 0.050 | 0.050 | 1.570 |

**Quality rules adopted** (chosen on training data, to be applied unchanged to
validation and test): keep a window only if observed span ≥ 4.5 s **and** max
internal gap ≤ 0.5 s.

Only **6 windows** fail — all subject 1649, Kicking Soccer Ball, at three time
positions, appearing in both watch-accel and watch-gyro. Consistent with a
watch-level recording interruption. **Zero phone-accel windows are affected.**

### Class support (training windows)

Well balanced: 5,537 (Typing) to 5,754 (Kicking Soccer Ball) windows per
activity; every activity has 40 or 41 training subjects. Class weighting judged
unnecessary; macro-averaged metrics still reported.

### Evaluation protocol

- 41 dev subjects → **5 fixed subject-wise folds** via `KFold(shuffle=True,
  random_state=42)` over the sorted subject array (equivalent to a grouped
  split; every window from a subject stays in one fold). Fold sizes 9/8/8/8/8.
- The exact mapping is persisted in `baseline/artifacts/development_folds.json`
  and must be reused by every later model comparison.
- All 18 activities are represented in every fold.
- **Primary model-selection metric: mean validation macro-F1 across folds.**
  Also reported: std across folds, balanced accuracy, accuracy, per-activity
  precision/recall/F1, row-normalized confusion matrix, per-subject performance.
- The 10 test subjects are **untouched** and remain reserved for a single final
  evaluation.
- Reusable helpers `calculate_metrics`, `summarize_oof_predictions`, and
  `generate_oof_predictions` operate on any model's out-of-fold prediction table,
  with asserts that train/validation subjects are disjoint per fold and that
  every window receives exactly one prediction.

### Initial phone-accelerometer feature table

**Phone-accelerometer only**, 25,811 windows (none lost to quality filtering),
3,709,121 sensor rows. **28 features** = 7 statistics (mean, std, min, max,
median, IQR, RMS) × 4 signals (x, y, z, magnitude where
magnitude = √(x²+y²+z²)). All finite, no missing values, unique window IDs.

Frequency-domain features were deliberately omitted because windows have
different sampling frequencies and have not been resampled.

### Initial phone-accelerometer results (out-of-fold, 5 subject folds)

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Majority class (`DummyClassifier`) | 0.0560 ± 0.0005 | 0.0556 ± 0.0000 | 0.0059 ± 0.0001 |
| **Random Forest** (300 trees, defaults, seed 42) | 0.3114 ± 0.0311 | 0.3126 ± 0.0300 | **0.3017 ± 0.0296** |

Per-fold RF macro-F1: 0.293, 0.333, 0.329, 0.292, 0.262. Fit+predict ~1.75 s per
fold.

**Per-activity F1 (RF, pooled out-of-fold):**

| Activity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Jogging | 0.894 | 0.913 | **0.903** | 1373 |
| Walking | 0.749 | 0.575 | 0.651 | 1444 |
| Stairs | 0.544 | 0.741 | 0.628 | 1409 |
| Kicking Soccer Ball | 0.560 | 0.622 | 0.589 | 1443 |
| Standing | 0.369 | 0.564 | 0.446 | 1447 |
| Playing Catch | 0.375 | 0.349 | 0.361 | 1443 |
| Folding Clothes | 0.354 | 0.341 | 0.348 | 1445 |
| Dribbling Basketball | 0.338 | 0.305 | 0.321 | 1444 |
| Clapping | 0.291 | 0.148 | 0.196 | 1443 |
| Eating Sandwich | 0.202 | 0.137 | 0.163 | 1444 |
| Eating Pasta | 0.147 | 0.171 | 0.158 | 1441 |
| Brushing Teeth | 0.143 | 0.166 | 0.154 | 1446 |
| Sitting | 0.111 | 0.200 | 0.143 | 1445 |
| Eating Chips | 0.118 | 0.092 | 0.103 | 1444 |
| Drinking from Cup | 0.095 | 0.097 | 0.096 | 1444 |
| Typing | 0.096 | 0.080 | 0.087 | 1371 |
| Eating Soup | 0.087 | 0.064 | 0.074 | 1444 |
| Writing | 0.084 | 0.062 | **0.071** | 1441 |

**Per-subject spread (41 held-out subjects):** macro-F1 mean 0.276, std 0.075,
min 0.122 (subject 1622), max 0.410. Weakest subjects: 1622, 1620, 1642, 1630,
1616.

**Confusion structure:** Walking↔Stairs overlap; Kicking Soccer Ball, Playing
Catch and Dribbling Basketball mutually confused; static, eating and fine-motor
activities scatter across many predicted classes rather than forming separable
groups.

### Four-stream Random Forest diagnostic

The identical quality rules, 28 statistical features, Random Forest
configuration, subject folds, and evaluation metrics were then applied to each
sensor stream independently.

| Stream | Valid windows | Mean macro-F1 | Fold std |
|---|---:|---:|---:|
| Phone accelerometer | 25,811 | 0.302 | 0.030 |
| Phone gyroscope | 25,622 | 0.268 | 0.018 |
| Watch accelerometer | 25,705 | **0.656** | 0.047 |
| Watch gyroscope | 25,585 | 0.529 | 0.041 |

Watch accelerometer is best for 17 of 18 activities; watch gyroscope is best
only for Dribbling Basketball. The watch advantage appears in every validation
fold, confirming that the weak phone-accelerometer result reflects sensor
placement and available signal rather than a failed modeling pipeline.

### Controlled watch accelerometer–gyroscope feature fusion

Valid watch windows were paired one-to-one by subject, activity, and ordinal
window position. All 25,585 valid gyroscope windows have an accelerometer
partner; 120 additional accelerometer windows were excluded so all comparison
models use the same population. All 41 development subjects remain represented.

The within-watch temporal alignment is adequate for whole-window feature
fusion: median gyroscope start offset is ~0.05 s, the 99th percentile is
~0.164 s, the maximum is ~0.40 s, and every pair shares at least 4.53 s of
coverage.

Each paired example contains 28 accelerometer and 28 gyroscope statistical
features. Three Random Forests were retrained on the exact paired population:

| Model | Features | Mean accuracy | Mean balanced accuracy | Mean macro-F1 |
|---|---:|---:|---:|---:|
| Watch accelerometer (paired) | 28 | 0.660 | 0.660 | 0.658 |
| Watch gyroscope (paired) | 28 | 0.534 | 0.534 | 0.529 |
| **Watch accel + gyro fusion** | 56 | **0.692** | **0.693** | **0.691** |

Fusion improves macro-F1 over paired accelerometer by ~0.033 and improves all
five validation folds. It improves 17 of 18 activities and 33 of 41 held-out
development subjects. The largest activity gains occur for Dribbling
Basketball, Playing Catch, Walking, Eating Pasta, and Stairs. Clapping is the
only declining activity, but remains strong at F1 0.845.

The fused pooled out-of-fold macro-F1 is 0.692. Eating Sandwich remains the
weakest class at F1 0.259. Subject-level macro-F1 ranges from 0.111 to 0.886,
with subjects 1637 and 1638 particularly difficult. The ten fixed test subjects
remain untouched.

**Baseline-stage benchmark:** watch accelerometer–gyroscope statistical feature
fusion with mean subject-fold macro-F1 **0.691 ± 0.048**.

---

## 7. Interpretation and open questions

### The central question

The stream diagnostic resolves the earlier concern about the low phone-only
score. Watch accelerometer reaches mean macro-F1 0.656 under the identical
pipeline, and adding watch gyroscope features raises it to 0.691. The pipeline
is behaving plausibly; modality and sensor placement explain much of the
original limitation.

The 20 Hz representation preserves the original classical result, and the
predeclared engineered feature groups raise mean macro-F1 to 0.7257. A compact
dual-stream CNN confirms that gyroscope information helps but reaches only
0.6762 across three seeds. On the same 222 features, histogram gradient
boosting reaches 0.7446 and improves all five folds over the Random Forest. A
fixed feature MLP reaches 0.7341 at seed 42, below both its predeclared seed-
repeat threshold and the boosting result. Histogram gradient boosting is
therefore the selected development winner.

### Known gaps

1. **Windowing is not gap-aware.** `windows_from_bout` bins by time offset within
   a bout and never treats gaps as segment boundaries, contradicting the EDA's
   own preprocessing requirement. Practical damage today is 6 windows, all
   caught by the post-hoc quality filter. It becomes a real problem the moment
   resampling or interpolation is introduced, because interpolation would bridge
   those gaps. Note this is part of the frozen team protocol.
2. **Temporal bandwidth remains a modeling assumption.** The 20 Hz artifact
   passes Random Forest parity for whole-window statistics, but high-rate
   streams are downsampled without an anti-alias filter. Parity does not prove
   that every short-duration or high-frequency pattern is retained for a
   sequence model.
3. **Cross-device fusion remains unsolved.** Joining phone and watch on the bout
   window index (`_0000` ↔ `_0000`) is cheap but assumes both devices started
   the bout simultaneously, which has not been verified. Late fusion avoids
   sample-level synchronization but still requires a justified mapping between
   phone and watch prediction units.
### Completed model-selection cycle

The frozen histogram-gradient-boosting pipeline has been fitted and evaluated
once on the reserved subjects. Aggregate, per-activity, confusion, and subject-
level results are recorded without post-test changes. The current test set is
closed to further model selection. Any future temporal modeling is a separate
research extension and requires a new independent test population for an
unbiased final comparison.

### Suggested discipline for `modified/`

`build_windowed_splits.py` already asserts that its output subject lists match
`split_meta.json`. Every future preprocessed artifact in that folder should
repeat that assertion and write its own `*_meta.json`. Large Parquet/NPZ/NPY
artifacts remain gitignored; small metadata JSON files remain tracked.

Deterministic resampled signals should be saved without population-level
normalization. Normalization or any other learned transformation must be fitted
inside each development fold and, for the final model only, refitted on all 41
development subjects before application to test subjects.

---

## 8. Environment

Python 3.12.1, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, matplotlib,
pyarrow, tqdm. Local `.venv` in the project root. macOS.

---

## 9. Business-driven grouped-target extension

This is a new decision problem built from the completed sensor and feature
pipeline, not a post-test revision of the original 18-class model. The accepted
business labels are:

| Business label | Original activities |
|---|---|
| `physical_activity` | Walking, Jogging, Stairs, Kicking Soccer Ball, Playing Catch, Dribbling Basketball |
| `food_or_drink` | Eating Soup, Eating Chips, Eating Pasta, Drinking from Cup, Eating Sandwich |
| `other_daily_activity` | Sitting, Standing, Typing, Brushing Teeth, Writing, Clapping, Folding Clothes |

On the 25,585 paired-watch development examples, the groups contain 8,470
physical-activity examples (33.11%), 7,105 food-or-drink examples (27.77%), and
10,010 other-daily-activity examples (39.12%). All three groups contain all 41
development subjects, so the controlled comparison used no class weighting.

`business_case/01_grouped_target.ipynb` reuses the canonical 5-second
paired-watch examples, deterministic 222-feature representation, frozen HGB
configuration, and exact five subject-wise development folds. It compares an
18-class model whose probabilities are summed into the three groups with the
same estimator trained directly on the grouped target:

| Approach | Mean macro-F1 | Fold std | Mean food precision | Mean food recall |
|---|---:|---:|---:|---:|
| Aggregated 18-class probabilities | 0.9123 | 0.0253 | 0.8808 | 0.8718 |
| **Direct 3-class model** | **0.9158** | 0.0262 | **0.9008** | 0.8647 |

The 0.0035 macro-F1 difference falls within the predeclared 0.01 practical-tie
range. Direct training is selected because it raises mean food-or-drink
precision by 0.0200, improves that metric in all five folds and 29 of 41
subjects, and is simpler and faster. Its pooled food-or-drink precision, recall,
and F1 are 0.9019, 0.8654, and 0.8833. The modest recall reduction is retained
as an explicit business trade-off.

Most errors occur between food-or-drink and other daily activity. Sitting is the
largest source of food false positives; Brushing Teeth is not. Physical activity
is well isolated, with pooled recall 0.9566. Subject performance remains uneven:
direct-model macro-F1 ranges from 0.4473 to 0.9984 with a median of 0.9414, and
subjects 1637 and 1638 are severe outliers under both compared approaches.

The selected direct-model out-of-fold probabilities are stored locally and feed
the downstream policy analysis. No model was fitted on all development subjects
in this notebook, and the window-level result is not a production notification
system.

`business_case/02_business_policy.ipynb` uses only these out-of-fold development
probabilities. It shows that the controlled-data food precision of 0.9019 would
fall to an estimated 0.5572 at an illustrative 5% food-window prevalence under
a simple label-shift assumption. A single five-second prediction is therefore
not accepted as a candidate event.

The frozen policy retains a food argmax only when its probability is at least
0.75 and requires four consecutive accepted windows, or 20 seconds of evidence.
Rejected food predictions fall back to the higher-probability non-food class; a
non-food prediction or missing window position resets the run. On development
data this policy has window-level food precision 0.9396 and recall 0.8210. It
detects 90.15% of controlled food recordings, alerts 2.46% of controlled
non-food recordings, and produces 0.6234 qualifying false runs per controlled
non-food hour.

The same final window labels also support movement accounting: every
`physical_activity` window contributes five seconds of movement evidence, with
no additional confidence or continuity requirement. This rule reaches pooled
physical precision 0.9798, recall 0.9569, and F1 0.9682. Across the controlled
development data it records 689.3 minutes versus 705.8 true physical minutes,
an aggregate underestimate of 2.34%. The median subject has no duration error,
but subject 1637 is underestimated by 45.1%, so these values are not validated
real-world active minutes or uniformly reliable wearer-level estimates.

These event diagnostics are not notification-performance claims. All recordings
are uniform 175-second single-activity blocks; cooldowns and daily alert limits
remain undefined. Subject 1637 has no food recording detected and subject 1638
has one of four detected. The 20-second requirement may also miss brief or
interrupted real-world intake behavior. Physical accumulation also does not
define exercise sessions or calorie expenditure. The complete schema-version-2
frozen contract is stored in `business_case/artifacts/business_policy.json`.

`business_case/03_test_evaluation.ipynb` verifies the frozen policy checksum,
fits the selected direct three-class HGB once on all 25,585 development
examples, and evaluates the complete pipeline once on 6,161 examples from the
ten reserved subjects. These subjects were not used to develop or evaluate the
grouped classifier or policies, but they had previously been evaluated for the
separate 18-class task and therefore were not globally unseen within the
repository.

| Final window result | Accuracy | Balanced accuracy | Macro-F1 | Food precision | Food recall | Physical recall |
|---|---:|---:|---:|---:|---:|---:|
| Raw three-class classifier | 0.8713 | 0.8643 | 0.8708 | 0.8671 | 0.7761 | 0.9129 |
| **Frozen window policy** | **0.8630** | **0.8508** | **0.8603** | **0.8981** | **0.7090** | **0.9133** |

The frozen food gate behaves as intended: relative to raw argmax it increases
food precision while reducing food recall. At recording level it detects 41 of
49 controlled food recordings (83.67%) and alerts on 5 of 127 controlled
non-food recordings (3.94%). It produces 12 false qualifying runs, equivalent
to 1.943 per controlled non-food hour. That rate is a controlled-recording
diagnostic, not an estimate of notifications in natural daily life.

The physical-accounting policy has precision 0.9799, recall 0.9133, and F1
0.9455. It records 157.75 minutes of movement evidence against 169.25 true
physical minutes, an aggregate underestimate of 11.5 minutes or 6.79%. This is
still not a validated active-minute, exercise-duration, or calorie measure.

Generalization varies substantially across subjects: subject-level macro-F1
ranges from 0.4620 to 0.9852, with a median of 0.9292. Subject 1640 has no food
recording detected and a 39.4% physical-duration underestimate; subjects 1607
and 1639 also show important food-policy errors. Subjects 1639 and 1640 were
also difficult in the separate 18-class evaluation, indicating persistent
wearer-level generalization risk.

`business_case/artifacts/grouped_test_result.json` is the tracked final record
and contains the metrics, limitations, artifact paths, and checksums. The model
and prediction tables remain local and gitignored. Its status is
`final_grouped_target_test_complete_no_post_test_changes`: the grouped test is
closed, and no classifier, threshold, or event-policy change may now be selected
from these results. Any further improvement requires a separately defined
development cycle and a new independent final-test population.
