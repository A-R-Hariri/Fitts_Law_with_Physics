# Fitts' Law Environment with Physics for Myoelectric Control

A configurable 2D Fitts' law target acquisition environment for evaluating EMG-driven myoelectric control. Designed around the Myo armband and LibEMG, it supports physics-based cursor dynamics, three task modes, hot-swappable classifiers, and per-frame CSV logging. Used for two distinct research purposes:

1. **EMG classifier evaluation**: comparing zero-shot cross-user models against within-user calibrated models in a standardized human-subject pointing task.
2. **Bypass prosthetic correlation research**: investigating the correlation between graphical cursor control performance and physical prosthetic hand control performance (screen-based Fitts vs. Dynamixel bypass arm).

---

## System Architecture

```
Myo armband (8-ch sEMG, 200 Hz)
    -> LibEMG online streamer
    -> OnlineEMGClassifier x3 (cross-user, within-CNN, within-MLP)
    -> UDP sockets (probabilities + velocity)
    -> input_thread (gesture -> 2D velocity vector)
    -> FittsTest (PySide6, QTimer at frame_rate Hz)
         -> physics integration (optional)
         -> cursor position update
         -> target hit detection
         -> per-frame CSV log
```

Three parallel classifiers run simultaneously on separate UDP ports. The active classifier is selected at runtime from the Dashboard without restarting the application.

---

## Files

```
Fitts_Law_with_Physics/
    main.py         -- Entry point: classifier loading, LibEMG setup, Dashboard launch
    Fitts.py        -- Dashboard (Qt config UI) and FittsTest (task window)
    collect.py      -- Per-user SGT data collection and within-user model training
    Replay.py       -- Frame-accurate replay of saved session CSV logs
    models.py       -- Model architecture definitions (CNN, CNN_GRL, MLP)
    utils.py        -- Hyperparameters, data loaders, training/evaluation utilities
```

---

## Dependencies

```
torch
PySide6
libemg
numpy
scikit-learn
```

```bash
pip install torch PySide6 libemg numpy scikit-learn
```

LibEMG requires additional setup for the Myo streamer. Refer to the [LibEMG documentation](https://libemg.github.io/libemg/) for OS-specific Myo driver installation.

---

## Task Modes

The environment implements three target acquisition modes, all selectable from the Dashboard at runtime.

**Mode A — Random**: on each acquisition, a new target appears at a random angle and random distance from the current cursor position. Distance is sampled uniformly from `target_distance_range` and target radius is drawn from `target_radius_list`. This mode does not enforce a fixed index of difficulty.

**Mode B — ISO Ring**: Has 12 targets which are placed at equal angular intervals on a circle of radius `ring_radius` centered on the screen. The sequence visits opposite sides of the ring following the standard alternating order `[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, ...]`. Multiple `(ring_radius, target_radius)` combinations can be provided as comma-separated lists in the Dashboard; the task cycles through all combinations in sequence. This mode produces the data needed for computing throughput (bits/s) under Fitts' law.

**Mode C — Moving Target**: the target moves continuously and bounces off the screen boundaries. Target velocity is set by the `c_vel` parameter and direction is randomized at initialization. The user must intercept the moving target and hold the cursor inside it for `hold_frames_required` frames.

---

## Physics Model

When physics is enabled, the cursor does not respond instantaneously to the classifier output. Instead, the input velocity signal is treated as a desired velocity, and a simple Newtonian integrator governs the actual cursor motion:

```
acc = (desired_v - actual_v) / mass
acc = clip(acc, -max_acceleration, +max_acceleration)
actual_v = (actual_v + acc) * damping
cursor_pos += actual_v
```

When disabled, `actual_v = desired_v` and the cursor moves directly proportional to the EMG velocity output with no inertia or delay.

The physics model is intended to study how inertia and damping affect Fitts' law throughput and error rate in EMG control, and to more closely replicate the dynamics of a physical prosthetic limb compared to a frictionless graphical cursor. This is the basis of the graphical vs. physical control correlation research.

---

## Gesture-to-Cursor Mapping

Four active gestures are mapped to a 2D cursor velocity vector. The `raw_velocity` from LibEMG's velocity estimator (proportional to EMG signal strength, clipped to [0, 1]) scales movement speed.

| Gesture | Direction | Notes |
|---------|-----------|-------|
| HC (hand close) | Down (+Y) | |
| HO (hand open) | Up (-Y) | |
| FX (flexion) | Left (−X) or Right (+X) | Controlled by flip_lr |
| EX (extension) | Right (+X) or Left (−X) | Controlled by flip_lr |
| NM (rest) | No movement | |

The `flip_lr` toggle swaps the FX/EX horizontal assignment to accommodate left-arm users. The `speed_multiplier` in the Dashboard scales the overall cursor speed by a constant factor on top of the EMG velocity signal.

---

## Running

### Step 1 — Collect per-user calibration data (within-user models only)

Skip this step if using only the cross-user model (`cnn_raw`).

```bash
python collect.py
```

This launches the LibEMG screen-guided training (SGT) GUI if calibration data does not already exist in `user_sgt/<NAME>/`. The GUI prompts the user to perform 15 repetitions of each gesture (3 s active, 2 s rest). After collection, `collect.py` automatically trains and saves six within-user models: CNN and MLP fine-tuned with 2, 5, and 13 reps.

The `NAME` variable in `utils.py` controls which subdirectory is used for a given participant. Change it before running for each new subject.

### Step 2 — Launch the environment

```bash
python main.py
```

On startup:
- All models listed in `model_names` are loaded from `pickles/` or `user_sgt/<NAME>/`
- Three LibEMG online classifiers start on ports 12346–12348
- Raw EMG is logged to `emg_logs/<NAME>/`
- The Dashboard window opens

### Step 3 — Configure and run a session

In the Dashboard:
1. Select the active model from the dropdown.
2. Set mode (A/B/C), frame rate, target parameters, and physics settings.
3. Optionally enter a label string that is appended to the log filename.
4. Check "Test Run" to prefix the output filename with `Test_` (useful for practice trials that should be excluded from analysis).
5. Click "Launch / Update Test" to open the task window.

The task window closes automatically when the target quota (`max_targets`) is reached or all ring/radius combinations are exhausted (Mode B). Logs are written to `fitts_logs/<NAME>/`.

---

## Data Collection and Logging

Each session produces a CSV file at `fitts_logs/<NAME>/[Test_]Fitts_<timestamp>_<model>[_<label>].csv`.

One row is written per frame at the configured frame rate. Columns:

| Column | Description |
|--------|-------------|
| `time` | Unix timestamp |
| `frame` | Frame index within session |
| `mode` | Task mode (A/B/C) |
| `model` | Active model name |
| `cursor_x`, `cursor_y` | Cursor position (pixels) |
| `target_x`, `target_y` | Target center (pixels) |
| `radius` | Target radius (pixels) |
| `X`, `Y` | Raw EMG velocity input (-1 to 1) |
| `vx`, `vy` | Actual cursor velocity after physics |
| `acc_x`, `acc_y` | Applied acceleration (0 if physics disabled) |
| `inside` | 1 if cursor inside target, 0 otherwise |
| `hold_count` | Consecutive frames inside target |
| `velocity` | Raw EMG velocity magnitude from LibEMG |
| `probs_0..4` | Classifier softmax probabilities per gesture |

The simultaneous EMG log at `emg_logs/<NAME>/` contains the raw 8-channel Myo stream, enabling post-hoc re-classification or offline analysis.

---

## Replay

To visually replay a saved session at the original frame rate:

```bash
python Replay.py
```

Edit the hardcoded CSV path in `Replay.py` before running (or adapt it to accept a command-line argument). The replay renders the same Qt window as the live session, correctly reconstructing the ring layout for Mode B from the logged target coordinates.

---

## Configuration Reference

All default parameters are defined in `utils.py` and can be overridden live from the Dashboard. The `PARAMS` dictionary passed to `SharedContext` on startup contains:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_rate` | 60 | Display and control loop update rate (Hz) |
| `mode` | `'B'` | Task mode (A/B/C) |
| `hold_frames_required` | 30 | Consecutive frames inside target to count as hit (modes A/B) |
| `target_timeout_frames` | 420 | Frames before target auto-advances (modes A/B) |
| `max_targets` | 12 | Targets to acquire before session ends |
| `target_radius_list` | `[20, 10]` | Target radii to cycle through (pixels) |
| `ring_radius_list` | `[300, 450]` | Ring radii for Mode B (pixels) |
| `target_distance_range` | `[200, 400]` | Random target distance range for Mode A (pixels) |
| `screen_size` | `(1690, 980)` | Task window dimensions (pixels) |
| `physics.enabled` | `False` | Enable inertia/damping on cursor |
| `physics.mass` | `5` | Inertia: higher = slower response |
| `physics.max_acceleration` | `0.08` | Max acceleration per frame |
| `physics.damping` | `1.0` | Velocity decay per frame (1.0 = no damping) |
| `c_vel` | `1` | Moving target speed for Mode C |
| `snap_back` | `False` | Teleport cursor to target on timeout |

---

## License

MIT — see [LICENSE](LICENSE).
