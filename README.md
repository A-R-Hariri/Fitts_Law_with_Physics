# Fitts' Law Target-Acquisition Environment for Myoelectric Control

A real-time 2D target-acquisition environment for evaluating EMG-driven cursor control with a Myo armband. The environment hosts several classifiers at once, switches between them at runtime without restarting, and logs one row per rendered frame.

This is the online evaluation platform used for the closed-loop study comparing calibration-free cross-user models against user-specific calibrated models. The offline pipeline that trains those models lives in a separate repository (see [Related repository](#related-repository)).

The optional physics layer (inertia, damping, acceleration limits) is included for work on how cursor dynamics affect throughput and error rate. It is disabled by default and was disabled for the reported study, so the cursor responds to the classifier output with no added dynamics.

---

## System overview

```
Myo armband (8 channels, 200 Hz, signed 8-bit)
  -> LibEMG streamer + OnlineDataHandler
  -> OnlineEMGClassifier x2 (cross-user port 12346, within-user port 12347)
       each wraps a MultiModelWrapper holding several models
  -> UDP on 127.0.0.1 (class probabilities + velocity)
  -> input_thread: predicted class -> 2D velocity vector, scaled by EMG amplitude
  -> SharedContext (multiprocessing Manager namespace)
  -> FittsTest window (PySide6 QTimer at frame_rate Hz)
       -> optional physics integration
       -> cursor update, hit and timeout detection
       -> per-frame CSV row
```

Classification runs at the LibEMG window rate (SEQ 40, INC 5, so 40 Hz). Rendering runs at 60 Hz and is decoupled from inference: each frame consumes the most recent decoded velocity rather than waiting for a new prediction, so the applied velocity is at most one classifier period old. No smoothing, majority vote, or rejection is applied.

Only two classifier processes run, one per input representation. Every raw-window model (all cross-user variants and the fine-tuned within-user models) is held inside the first wrapper; the hand-crafted-feature model is held inside the second. Switching models is a dictionary lookup inside the wrapper keyed on `SharedContext.active_model_name`, so there is no reload cost and no gap in the stream.

---

## Repository contents

```
Fitts_Law_with_Physics/
  main.py               Entry point: model loading, LibEMG setup, UDP sockets, Dashboard launch
  fitts.py              Dashboard (Qt config window) and FittsTest (task window, logging)
  collect.py            Screen-guided training data collection and within-user model training
  models.py             Architectures: MHCNN, CNN, CNN_HCF, MLP, GRL variants, RunningNorm
  utils.py              Participant ID, paths, task parameters, feature settings, shared utilities
  Replay.py             Frame-accurate replay and figure export from a saved session log
  fitts_analysis.py     Trial extraction, metrics, planned contrasts, plots for a whole study
  eval_sgt_cross.py     Offline check of cross-user checkpoints on the participant's own SGT data
  mouse_cursor_control.py   Standalone system-mouse control using the same classifier stack
```

Logs, checkpoints, gesture images, and figures are excluded from version control.

---

## Requirements

```
torch
PySide6
libemg
numpy
pandas
scipy
scikit-learn
matplotlib
```

```bash
pip install torch PySide6 libemg numpy pandas scipy scikit-learn matplotlib
```

LibEMG needs a working Myo streamer. Driver setup is OS specific; see the [LibEMG documentation](https://libemg.github.io/libemg/).

A CUDA device is assumed (`DEVICE = 'cuda'` in `utils.py`). CPU inference is supported by changing that constant.

---

## Configuration

All participant and task settings live in `utils.py`.

`NAME` is the participant identifier. It selects three directories and must be changed before each new participant:

```
user_sgt/<NAME>/     calibration data and within-user weights
emg_logs/<NAME>/     raw 8-channel EMG stream
fitts_logs/<NAME>/   per-frame task logs
```

Cross-user checkpoints and the population velocity thresholds are read from `checkpoints/`.

`PARAMS` holds the task defaults. Every entry except `screen_size` and the radius lists can also be changed live from the Dashboard.

| Parameter | Default | Meaning |
|---|---|---|
| `frame_rate` | 60 | Render and control loop rate (Hz) |
| `mode` | `'B'` | Task mode (A, B, or C) |
| `hold_frames_required` | 45 | Dwell frames inside the target to register a hit (0.75 s at 60 Hz) |
| `target_timeout_frames` | 480 | Frames before the trial is abandoned (8 s at 60 Hz) |
| `max_targets` | 12 | Acquisitions per ring/radius combination |
| `ring_radius_list` | `[300, 375, 450]` | Mode B ring radii (px) |
| `target_radius_list` | `[30, 21, 12]` | Target radii (px), paired elementwise with the ring radii |
| `target_distance_range` | `[200, 400]` | Mode A distance range (px) |
| `screen_size` | `(1690, 980)` | Task window size (px) |
| `physics.enabled` | `False` | Enable inertia and damping |
| `physics.mass` | `5` | Higher mass gives a slower response |
| `physics.max_acceleration` | `0.08` | Acceleration clip per frame |
| `physics.damping` | `1.0` | Velocity retained per frame (1.0 is no damping) |
| `c_vel` | `1` | Mode C target speed |
| `snap_back` | `False` | Teleport the cursor onto the target after a timeout |

`VEL_CONSTANT` in `fitts.py` is the base gain, 20 px per frame at full EMG amplitude. The Dashboard speed multiplier scales it further and is set automatically on model change: 1.2 for cross-user models, 1.0 for within-user models. The cross-user boost was fixed in piloting so that cursor speed was comparable across paradigms and the comparison would not be dominated by the velocity mapping.

---

## Difficulty conditions

In Mode B the ring and target radius lists are zipped, so the defaults give three conditions run in sequence, each for `max_targets` acquisitions. Amplitude is the ring diameter and width is the target diameter, giving the Shannon index of difficulty `ID = log2(A/W + 1)`:

| Condition | Ring radius (px) | A (px) | W (px) | ID (bits) |
|---|---|---|---|---|
| Easy | 300 | 600 | 60 | 3.46 |
| Medium | 375 | 750 | 42 | 4.24 |
| Hard | 450 | 900 | 24 | 5.27 |

Twelve targets are placed at equal angles on each ring and visited in the standard multidirectional order `[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, ...]`, so consecutive targets sit on opposite sides of the ring. With the defaults, one model produces 36 acquisitions.

---

## Task modes

**Mode A, random.** Each new target appears at a random angle and a random distance drawn from `target_distance_range`, with a radius drawn from `target_radius_list`. Index of difficulty is not fixed, so this mode is for practice and familiarisation rather than throughput measurement.

**Mode B, ISO ring.** The multidirectional arrangement described above. This is the mode used for all reported results.

**Mode C, moving target.** The target translates at `c_vel` and reflects off the window edges. The participant must intercept it and hold the cursor inside for the dwell period. Intended for pursuit and interception work, not for Fitts' law measurement.

---

## Control mapping

The predicted class sets the direction and the LibEMG velocity estimate sets the speed. Rest holds the cursor still.

| Class | Cursor direction |
|---|---|
| NM (no motion) | stationary |
| HC (hand close) | down |
| HO (hand open) | up |
| FX (wrist flexion) | left, or right if `flip_lr` is set |
| EX (wrist extension) | right, or left if `flip_lr` is set |

`flip_lr` swaps the horizontal assignment for participants wearing the band on the other arm.

Velocity scaling is set per paradigm:

- Within-user models call `add_velocity` on the participant's own calibration windows, which is standard LibEMG behaviour.
- Cross-user models load population thresholds from `checkpoints/th_max_dic.npy` and `checkpoints/th_min_dic.npy`, computed offline over the training users. No participant data is used, so the cross-user path stays calibration free end to end.

---

## Physics model

With physics enabled the classifier output is treated as a desired velocity and the cursor follows a first-order integrator:

```
acc = (desired_v - actual_v) / mass
acc = clip(acc, -max_acceleration, +max_acceleration)
actual_v = (actual_v + acc) * damping
pos += actual_v
```

With physics disabled, `actual_v = desired_v` and position is integrated directly. The logged `acc_x` and `acc_y` columns are zero in that case.

---

## Running a session

### 1. Collect calibration data (only if within-user models are needed)

```bash
python collect.py
```

If `user_sgt/<NAME>/` is empty this launches the LibEMG screen-guided training GUI: `TOTAL_REPS` repetitions of each of the five gestures, 3 s of contraction and 3 s of rest per repetition. It then trains and saves the within-user models, using the first 1 and the first 5 repetitions so that both calibration budgets are available online.

Skip this step for a purely calibration-free session.

### 2. Launch

```bash
python main.py
```

On startup `main.py`:

- loads every checkpoint listed in `model_names` (cross-user weights from `checkpoints/`, within-user weights from `user_sgt/<NAME>/`),
- shuffles that list, so the order in the Dashboard dropdown differs per participant,
- starts the two online classifiers and binds their UDP sockets,
- starts logging the raw EMG stream to `emg_logs/<NAME>/`,
- opens the Dashboard.

The dropdown shows one-letter codes from `MODEL_DICT` rather than model names. Combined with the shuffle, this keeps the model identity out of view during a session and keeps the presentation order counterbalanced.

### 3. Run

In the Dashboard: pick the model code, set mode and task parameters, optionally type a label to append to the filename, then click **Launch / Update Test**.

Keep **Test Run** checked for practice blocks. It prefixes the filename with `Test_`, and both `Replay.py` and `fitts_analysis.py` skip any file or subject folder whose name contains `test`.

The task window closes on its own once every ring/radius combination has reached `max_targets`.

---

## Log format

One CSV per session at:

```
fitts_logs/<NAME>/[Test_]Fitts_<YYYY-MM-DD_HH-MM-SS>_<model>[_<label>].csv
```

One row per rendered frame:

| Column | Content |
|---|---|
| `time` | Unix timestamp |
| `frame` | Frame index within the session |
| `mode` | Task mode |
| `model` | Active model name |
| `cursor_x`, `cursor_y` | Cursor position (px) |
| `target_x`, `target_y` | Target centre (px) |
| `radius` | Target radius (px) |
| `X`, `Y` | Decoded input velocity, range -1 to 1 |
| `vx`, `vy` | Applied cursor velocity after physics |
| `acc_x`, `acc_y` | Applied acceleration, 0 when physics is off |
| `inside` | 1 when the cursor is inside the target |
| `hold_count` | Consecutive frames inside the target |
| `velocity` | EMG amplitude estimate from LibEMG |
| `probs_0` to `probs_4` | Class probabilities, ordered NM, HC, FX, EX, HO |

The parallel raw EMG log under `emg_logs/<NAME>/` carries the full 8-channel stream, which allows any session to be reclassified offline without rerunning the participant.

Trials are recovered from the log rather than written explicitly: a new trial starts wherever the target coordinates or radius change, conditions are grouped by quantised target distance, and the outcome is a hit if the trial ended on a full dwell and a timeout otherwise.

---

## Replay

```bash
python Replay.py
```

Set `USER_ID`, `MODEL`, and `SESSION_INDEX` at the top of the file, or point `LOG_PATH` at a specific CSV. The window reconstructs the ring layout and cursor motion from the logged coordinates alone.

Controls: space pauses, arrows step one frame, `,` and `.` step by trial, `[` and `]` step by condition, `T` toggles the trace, `S` saves a frame.

The side panel exposes the trace options: scope (session, condition, trial, or sliding window), and colour source (speed, gesture, elapsed time, or flat). Speed colouring auto-fits its upper limit to the 98th percentile of the session, which must be pinned with `SPEED_VMAX` when producing figures that are compared across models.

`PAPER_MODE` switches to a white background and `SHOT_SCALE` controls export resolution. **Save frame** writes the current state and **Save full path** writes the whole session trace, both at export resolution rather than screen resolution.

---

## Analysis

```bash
python fitts_analysis.py
```

Walks the log root, skips anything marked as a test, extracts trials, and writes one output directory containing:

| File | Content |
|---|---|
| `manifest.csv` | One row per discovered log file |
| `trials_long.csv` | One row per trial |
| `per_subject_model.csv` | Metrics per subject and model |
| `per_subject_model_condition*.csv` | The same, split by difficulty condition |
| `fitts_regression.csv` | Slope, intercept, grand and per-subject R2 for MT against ID |
| `summary_by_model.csv` | Mean, SD, median, IQR, range, plus rank by mean and by consistency |
| `contrasts_<metric>.csv` | Planned contrasts per metric |
| `*.png` | Per-metric boxplots with subject-coloured points, a combined grid, by-ID panels, and the regression panels |

Metrics computed per trial and averaged per subject and model: completion rate; movement time excluding the dwell period; nominal, effective, and mean-of-means throughput, each with a penalized variant that charges failed trials their full timeout; path efficiency; direction-change ratio; overshoots; stopping distance; reaction time; and a quality composite that z-scores the four trajectory-quality measures across all subject-model cells, flips signs so higher is better, and averages them.

Two pre-specified contrast families, both paired within subject and Holm corrected inside the family, reported with Wilcoxon signed-rank statistics, matched-pairs rank-biserial correlation, and Cohen's dz:

- **Family A**: each single-intervention cross-user model against the cross-user baseline.
- **Family B**: pooled cross-user against pooled calibrated, one pair per subject.

Model order, display names, and colours are fixed in the constants at the top of the file, so figures stay comparable across runs, and logs from any version of the environment can be pooled in a single run.

---

## Offline check

```bash
python eval_sgt_cross.py
```

Runs the cross-user checkpoints over the calibration data collected by `collect.py` and reports accuracy, active accuracy, balanced accuracy, and macro F1 for each participant, with and without streaming normalization and with and without active-region segmentation. Useful as a sanity check on band placement and signal quality before an online session, and for relating a participant's offline separability to their online performance.

---

## Related repository

The offline pipeline that produces the cross-user checkpoints, including dataset preparation, architecture and feature comparisons, loss variants, and the within-user references, is in `EPN612_Cross_User`.

---

## License

MIT. See `LICENSE`.