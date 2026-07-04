"""
Fitts' law analysis for online (real-time) myoelectric model evaluation.

Reads per-subject, per-model ISO 9241-9 (Mode B, circular ring) cursor logs and
computes the standard online myoelectric control metrics (throughput, path
efficiency, completion rate, overshoot, reaction time, stopping distance),
validates the Fitts task via the movement-time vs index-of-difficulty
regression, runs repeated-measures statistics across models (Friedman, pairwise
Wilcoxon with Holm-Bonferroni correction, Cohen's dz, matched-pairs
rank-biserial), and renders per-metric boxplots with per-subject jitter.

LOG FORMAT (post-fix runner): the dwell counter increments while the cursor
holds inside the target. The runner logs the dwell-completion (firing) frame in
the trial's OWN block, with hold_count == HOLD_FRAMES and the trial's own target
coordinates; the target advances on the FOLLOWING frame. Acquisition is flagged
at hold_count >= HOLD_FRAMES. This differs from the pre-fix pilot runner, which
logged the firing frame after the target had already switched (it landed as the
first row of the next block). A guard in segment_trials raises if a pre-fix log
is passed in, so the two formats are never silently mixed.

Metric definitions follow Scheme and Englehart 2013 (IEEE TNSRE), Wurth and
Hargrove 2014 (J NeuroEng Rehabil), Eddy et al. 2023 (LibEMG, IEEE Access), and
Waris et al. 2018 / 2020. Effective throughput follows the ISO 9241-9
effective-width method (Soukoreff and MacKenzie 2004).

The unit of analysis is the subject. Higher mean with lower across-subject
spread is treated as the target outcome.

Notation per trial: D = straight-line distance from the start cursor position to
the target center; W = target width (diameter); t_acq = time of the
dwell-completion (firing) frame; t_start = time the target appeared; t_end =
time of the last logged frame; DWELL = HOLD_FRAMES / FRAME_RATE (the mandatory
hold); P = total cursor path length; d_eff = straight-line distance from the
start cursor to the final cursor position (at acquisition or at timeout);
ID = log2(D / W + 1).

Metrics:
- completion_rate: percent of targets acquired within the timeout. = 100 * n_success / n_trials. All trials.
- movement_time (MT): reaching time from target onset to acquisition, mandatory dwell removed. = t_acq - t_start - DWELL, t_acq at the firing frame (hold_count == HOLD_FRAMES). Success only.
- movement_time_penalized: MT for successes (dwell removed); full elapsed time for failures. = (t_acq - t_start - DWELL) if success else (t_end - t_start). All trials.
- throughput_nominal (TP): Shannon throughput, mean of per-trial ID/MT. = mean(ID / MT). Success only.
- throughput_nominal_penalized: mean of ID / MT_penalized over all trials (failures charged their timeout). All trials.
- throughput_meanofmeans: per-condition mean ID over mean MT, averaged across conditions. = mean_c( mean(ID)_c / mean(MT)_c ). Success only.
- throughput_meanofmeans_penalized: same with MT_penalized over all trials. All trials.
- throughput_effective: ISO effective throughput, mean_c( IDe_c / mean(MT)_c ), IDe = log2(Ae / We + 1), Ae = mean amplitude in the condition,
    We = 4.133 * SDx. SDx is the std of selection endpoints projected onto the start->target axis, computed PER CONDITION (the axis and amplitude differ
    by condition). 4.133 = 2 * 2.0665 brackets the central 96% of a Gaussian endpoint spread, so We is the width that would have yielded a 4% miss rate
    given the observed precision. Success only.
- throughput_effective_penalized: same IDe but divided by mean MT_penalized over all trials in the condition. All trials in denominator; We from successes.
- path_efficiency (PE): directness of the trajectory, start to final position. = 100 * d_eff / P, capped at 100. All trials.
- direction_change_ratio: cardinal movement segments taken divided by the minimum needed. = N_seg / N_seg_min, N_seg_min = 1 if the target is on a cardinal axis from the start else 2. Optimal = 1.0. All trials.
- overshoots: count of target-boundary exits after first entry (enter-then-leave episodes). Success only.
- stopping_distance: maximum penetration past the target boundary after first entry, in px. Success only.
- reaction_time: time from target onset to first cursor movement above a step threshold. Success only.
"""

import os
import re
import math
import warnings
from os.path import join, isdir, isfile
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

warnings.filterwarnings("ignore")


# ======== CONFIG ========

FITTS_ROOT = "fitts_logs"
OUT_DIR = "fitts_results"

# Task parameters must match the run that produced the logs. Pulled from utils so
# the analysis stays locked to the runner config; falls back to literals when the
# training package is not importable on the analysis machine.
try:
    from utils import PARAMS
    FRAME_RATE = PARAMS['frame_rate']
    HOLD_FRAMES = PARAMS['hold_frames_required']
    TIMEOUT_FRAMES = PARAMS['target_timeout_frames']
    SCREEN_SIZE = PARAMS['screen_size']
    RING_RADII = PARAMS['ring_radius_list']
    TARGET_RADII = PARAMS['target_radius_list']
except Exception:
    FRAME_RATE = 60
    HOLD_FRAMES = 30
    TIMEOUT_FRAMES = 420
    SCREEN_SIZE = (1690, 980)
    RING_RADII = [300, 450]
    TARGET_RADII = [20, 10]

# Screen center. None infers it per file from the first logged cursor position
# (the cursor always starts at screen center). Fallback is SCREEN_SIZE / 2.
SCREEN_CENTER = None

# The task presents (n_rings x n_widths) conditions x max_targets trials per
# (subject, model). No targets dropped by default. Set DROP_FIRST_PER_CONDITION
# to drop the first target of each condition (ISO multidirectional convention).
DROP_FIRST_MOVE = False
DROP_FIRST_PER_CONDITION = False
MIN_TRIAL_FRAMES = 5  # blocks shorter than this are pure acquisition artifacts
FIX_SORT = True

# Quality metrics that require reaching the target (overshoot, stopping
# distance, reaction time) and the success-only throughput / MT use acquired
# trials. Completion, the penalized metrics, path efficiency, and the
# direction-change ratio use all trials.
MOVE_STEP_PX = 1.0          # cursor step magnitude that counts as movement onset
DWELL_TIME = HOLD_FRAMES / FRAME_RATE
MIN_TRIALS_FOR_WE = 3       # min successful trials per condition for a usable We (SDx needs >= 2; 3 keeps it stable)
DIR_MIN_STEP_PX = 2.0       # per-frame step below this is ignored for direction
DIR_MERGE_PX = 15.0         # a cardinal segment shorter than this (px) is treated as jitter and merged
# Minimum valid movement time after dwell subtraction. Anything shorter means
# the cursor was already inside the target when it appeared (trivial acquisition:
# no real Fitts movement occurred). These trials count for completion rate but
# are excluded from MT and throughput averages to prevent ID/~0 TP spikes.
MT_MIN_VALID = 2.0 / FRAME_RATE   # 2 frames

# Direction of improvement, used only for sorting and reporting.
HIGHER_IS_BETTER = {
    "throughput_nominal": True,
    "throughput_meanofmeans": True,
    "throughput_effective": True,
    "path_efficiency": True,
    "completion_rate": True,
    "movement_time": False,
    "reaction_time": False,
    "overshoots": False,
    "stopping_distance": False,
    "direction_change_ratio": False,
    # Penalized variants (failed trials folded in).
    "throughput_nominal_penalized": True,
    "throughput_meanofmeans_penalized": True,
    "throughput_effective_penalized": True,
    "movement_time_penalized": False,
}

# Metrics shown in the headline boxplot grid. Each penalized metric is placed
# next to its success-only counterpart.
PLOT_METRICS = [
    "throughput_nominal",
    "throughput_nominal_penalized",
    "movement_time",
    "movement_time_penalized",
    "throughput_effective",
    "throughput_effective_penalized",
    "throughput_meanofmeans",
    "throughput_meanofmeans_penalized",
    "completion_rate",
    "path_efficiency",
    "direction_change_ratio",
    "overshoots",
    "stopping_distance",
    "reaction_time",
]

# Fixed left-to-right model order for every boxplot, independent of metric value,
# so a model sits in the same x-slot in every panel. Present models not listed
# here are appended (alphabetically) rather than dropped.
MODEL_ORDER = [
    "cross_mhcnn_raw_trp",
    "cross_mhcnn_raw_rest",
    "cross_mhcnn_raw_1va",
    "cross_mhcnn_raw_base",
    "cross_mhcnn_raw_base-rn",
    "cross_mhcnn_segmented_base",
    "within_cnnhcf_raw_base-5",
    "within_mhcnn_raw_base-ft-1",
    "within_mhcnn_raw_base-ft-5",
]

# Short tick labels so the x-axis is not a wall of long log names. Edit freely;
# any model missing here falls back to its raw log name.
DISPLAY_NAMES = {
    "within_mhcnn_raw_base-ft-5": "FT-5",
    "within_mhcnn_raw_base-ft-1": "FT-1",
    "within_cnnhcf_raw_base-5":   "Within-5",
    "cross_mhcnn_segmented_base": "Segmented",
    "cross_mhcnn_raw_base-rn":    "RunNorm",
    "cross_mhcnn_raw_base":       "Base",
    "cross_mhcnn_raw_1va":        "Contrastive",
    "cross_mhcnn_raw_rest":       "Rest",
    "cross_mhcnn_raw_trp":        "Triplet",
}


# ======== LOG DISCOVERY AND LOADING ========

COLUMNS = ["time", "frame", "mode", "model", "cursor_x", "cursor_y",
           "target_x", "target_y", "radius", "X", "Y", "vx", "vy",
           "acc_x", "acc_y", "inside", "hold_count", "velocity",
           "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]

NUM_COLS = ["time", "frame", "cursor_x", "cursor_y", "target_x", "target_y",
            "radius", "X", "Y", "vx", "vy", "acc_x", "acc_y", "inside",
            "hold_count", "velocity"]


def _is_test_name(name):
    return 'test' in name.lower()


def discover_logs(root):
    """Return list of (subject_id, model_name, filepath) tuples."""
    found = []
    if not isdir(root):
        raise FileNotFoundError("Log root not found: %s" % root)
    for subj in sorted(os.listdir(root)):
        subj_path = join(root, subj)
        if not isdir(subj_path) or _is_test_name(subj):
            continue
        for fname in sorted(os.listdir(subj_path)):
            if not fname.lower().endswith(".csv") or _is_test_name(fname):
                continue
            found.append((subj, fname, join(subj_path, fname)))
    return found


def _model_from_filename(fname):
    # Fitts_YYYY-MM-DD_HH-MM-SS_<model>.csv
    stem = re.sub(r"\.csv$", "", fname, flags=re.IGNORECASE)
    m = re.match(r"^Fitts_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$", stem)
    return m.group(1) if m else stem


def load_log(filepath):
    """Load one CSV log into a numeric DataFrame, sorted by frame."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "cursor_x", "cursor_y", "target_x", "target_y"])
    df = df.sort_values("frame").reset_index(drop=True)
    return df


# ======== TRIAL SEGMENTATION ========

def _infer_center(df):
    if SCREEN_CENTER is not None:
        return float(SCREEN_CENTER[0]), float(SCREEN_CENTER[1])
    return float(df["cursor_x"].iloc[0]), float(df["cursor_y"].iloc[0])


def _cardinal_segments(cx, cy, min_step=DIR_MIN_STEP_PX, merge_px=DIR_MERGE_PX):
    """
    Number of cardinal (Manhattan) movement segments in a cursor trajectory.

    The control maps gestures to up/down/left/right, so each instant of motion is
    quantized to the dominant cardinal axis and sign. Consecutive frames in the
    same cardinal direction form one segment; a new segment starts on every
    direction change. Per-frame steps below min_step are ignored (no motion), and
    a segment whose total displacement is below merge_px is treated as jitter and
    removed before counting. Returns the segment count (0 if the cursor never
    moved). The user-facing "direction change" count equals this segment count.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    if cx.size < 2:
        return 0
    dx = np.diff(cx)
    dy = np.diff(cy)
    runs = []  # [label, accumulated_displacement]
    for sx, sy in zip(dx, dy):
        mag = math.hypot(sx, sy)
        if mag < min_step:
            continue
        if abs(sx) >= abs(sy):
            lab = 0 if sx > 0 else 1
        else:
            lab = 2 if sy > 0 else 3
        if runs and runs[-1][0] == lab:
            runs[-1][1] += mag
        else:
            runs.append([lab, mag])
    if not runs:
        return 0
    kept = [r for r in runs if r[1] >= merge_px] or runs
    merged = []
    for lab, d in kept:
        if merged and merged[-1][0] == lab:
            merged[-1][1] += d
        else:
            merged.append([lab, d])
    return len(merged)


def segment_trials(df, subject, model):
    """
    Split one session into trials and compute per-trial metrics.

    A trial is a maximal run of rows sharing the same (target_x, target_y). The
    dwell counter increments while the cursor holds inside the target; the
    post-fix runner logs the dwell-completion (firing) frame in this trial's own
    block as hold_count == HOLD_FRAMES with the trial's own target, then advances
    the target on the following frame. Acquisition is flagged at
    hold_count >= HOLD_FRAMES. No look-ahead to the next block is used.

    A format guard raises if a pre-fix (pilot) log is passed in: there the firing
    frame landed on the next block (target already advanced), so a block would
    begin on hold_count >= HOLD_FRAMES.
    """
    cx, cy = _infer_center(df)

    tx = df["target_x"].to_numpy()
    ty = df["target_y"].to_numpy()
    changed = (np.diff(tx) != 0) | (np.diff(ty) != 0)
    block_id = np.concatenate([[0], np.cumsum(changed)])
    df = df.assign(_block=block_id)

    blocks = [g for _, g in df.groupby("_block", sort=True)]

    # Format guard. Post-fix blocks start with the cursor outside (or just
    # entering) the new target, so hold_count begins at 0 or 1. A block that
    # begins on hold_count >= HOLD_FRAMES is the carried firing-frame artifact of
    # the pre-fix runner and must be analysed with the pilot handler.
    first_holds = np.array([float(g["hold_count"].iloc[0]) for g in blocks])
    if np.any(first_holds >= HOLD_FRAMES):
        raise ValueError(
            "Pre-fix (pilot) log format detected for subject=%s model=%s: a block "
            "begins on hold_count >= HOLD_FRAMES (carried firing-frame artifact). "
            "This analysis targets post-fix logs; use the pilot-tagged version for "
            "pilot data." % (subject, model))

    rows = []

    # Within-condition running index, to flag the first trial of each
    # contiguous condition run.
    prev_cond = None
    cond_run_idx = -1

    for bi, blk in enumerate(blocks):
        blk = blk.reset_index(drop=True)
        n = len(blk)
        if n < MIN_TRIAL_FRAMES:
            continue  # pure acquisition artifact, not a real trial

        t_x = float(blk["target_x"].iloc[0])
        t_y = float(blk["target_y"].iloc[0])
        radius = float(blk["radius"].iloc[0])
        width = 2.0 * radius

        start = (float(blk["cursor_x"].iloc[0]), float(blk["cursor_y"].iloc[0]))

        # Task distance: start cursor to target center (drives ID). Not called
        # amplitude here to keep the ID input unambiguous.
        distance = math.hypot(start[0] - t_x, start[1] - t_y)
        ring_raw = math.hypot(t_x - cx, t_y - cy)
        ring = min(RING_RADII, key=lambda r: abs(r - ring_raw)) if RING_RADII else round(ring_raw)
        condition = "ring%d_w%d" % (int(ring), int(width))

        if condition != prev_cond:
            cond_run_idx += 1
            is_first_in_cond = True
        else:
            is_first_in_cond = False
        prev_cond = condition

        # Outcome via the dwell counter. A dwell completion fires at
        # hold_count == HOLD_FRAMES, logged in this block with the correct target.
        hold_arr = blk["hold_count"].to_numpy().astype(float)
        acq_rows = np.where(hold_arr >= HOLD_FRAMES)[0]
        success = acq_rows.size > 0
        acq_i = int(acq_rows[-1]) if success else (n - 1)
        end_i = acq_i

        # Selection point: cursor where the dwell completed (success) or where it
        # sat at timeout (failure).
        end = (float(blk["cursor_x"].iloc[end_i]), float(blk["cursor_y"].iloc[end_i]))

        timed_out = (not success) and (n >= TIMEOUT_FRAMES * 0.9)

        t_start = float(blk["time"].iloc[0])
        t_end = float(blk["time"].iloc[-1])
        # The firing frame (hold_count == HOLD_FRAMES) is the selection instant.
        # Adding one frame converts the discrete firing sample to the continuous
        # fire time, so subtracting the nominal dwell (HOLD_FRAMES / FRAME_RATE)
        # leaves exactly the onset->reach time. The +1 frame is below the noise
        # floor (one frame on multi-second MTs) but keeps the dwell cancelling
        # cleanly. Failures never dwell; their penalized time is the full elapsed
        # duration with nothing subtracted.
        t_acq = (float(blk["time"].iloc[acq_i]) + 1.0 / FRAME_RATE) if success else np.nan
        mt_raw = (t_acq - t_start - DWELL_TIME) if success else np.nan
        # Trivial acquisition: cursor was already inside the target when it
        # appeared (e.g. it drifted there during a preceding timeout). The hold
        # counter reaches HOLD_FRAMES within one dwell window, giving near-zero or
        # negative MT after dwell subtraction. Exclude from MT/TP averages so that
        # ID/~0 cannot spike the mean-of-ratios, while keeping success=1 so the
        # trial still counts toward completion rate.
        if success and (np.isnan(mt_raw) or mt_raw < MT_MIN_VALID):
            mt = np.nan
        else:
            mt = mt_raw
        mt_penalized = mt if success else max(t_end - t_start, 1.0 / FRAME_RATE)

        # Path from the start to the selection point (acquisition or timeout).
        seg_x = blk["cursor_x"].to_numpy()[:end_i + 1]
        seg_y = blk["cursor_y"].to_numpy()[:end_i + 1]
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        path_len = float(np.hypot(dx, dy).sum())
        # Path efficiency: directness from the start cursor to the final cursor
        # position, over the whole path. One definition for success and failure.
        # 100 percent is a perfectly straight path.
        eff_disp = math.hypot(end[0] - start[0], end[1] - start[1])
        pe = (eff_disp / path_len) if path_len > 1e-9 else np.nan
        if pe is not None and not np.isnan(pe):
            pe = min(pe, 1.0)

        # Direction-change ratio over the same start->selection path: cardinal
        # movement segments taken versus the minimum needed. Under cardinal
        # control a target off both axes needs at least two segments (one per
        # axis); a target aligned with an axis (smaller offset within one target
        # radius) needs one. Computed for every trial, hit or miss.
        n_seg = _cardinal_segments(seg_x, seg_y)
        off_x = abs(t_x - start[0])
        off_y = abs(t_y - start[1])
        min_seg = 1 if min(off_x, off_y) <= radius else 2
        dir_ratio = (n_seg / min_seg) if n_seg > 0 else np.nan

        # Overshoot count: number of inside 1 -> 0 transitions within the block
        # (each is an enter-then-leave before final acquisition).
        inside = blk["inside"].to_numpy().astype(int)
        leaves = int(np.sum((inside[:-1] == 1) & (inside[1:] == 0))) if n > 1 else 0
        overshoots = leaves

        # Stopping distance: maximum penetration beyond the target boundary
        # after first entry (px). 0 if it never strays back out after entering.
        entry_idx = np.argmax(inside == 1) if inside.any() else None
        if entry_idx is not None and inside.any():
            post = blk.iloc[entry_idx:]
            dcent = np.hypot(post["cursor_x"].to_numpy() - t_x,
                             post["cursor_y"].to_numpy() - t_y)
            stop_dist = float(np.max(np.clip(dcent - radius, 0.0, None)))
        else:
            stop_dist = np.nan

        # Reaction time: time from trial start to first cursor movement.
        step = np.hypot(dx, dy) if n > 1 else np.array([])
        moved = np.where(step > MOVE_STEP_PX)[0]
        if moved.size > 0:
            onset_row = moved[0] + 1
            rt = float(blk["time"].iloc[onset_row] - blk["time"].iloc[0])
        else:
            rt = np.nan

        id_nominal = math.log2(distance / width + 1.0) if width > 0 and distance > 0 else np.nan

        rows.append({
            "subject": subject,
            "model": model,
            "trial_global": bi,
            "condition": condition,
            "ring_radius": int(ring),
            "target_radius": radius,
            "width": width,
            "is_first_in_condition": is_first_in_cond,
            "n_frames": n,
            "distance": distance,
            "id_nominal": id_nominal,
            "start_x": start[0], "start_y": start[1],
            "target_x": t_x, "target_y": t_y,
            "end_x": end[0], "end_y": end[1],
            "success": int(success),
            "timed_out": int(timed_out),
            "movement_time": mt,
            "mt_penalized": mt_penalized,
            "path_length": path_len,
            "path_efficiency": (pe * 100.0) if pe is not None and not np.isnan(pe) else np.nan,
            "n_movement_segments": n_seg,
            "min_movement_segments": min_seg,
            "direction_change_ratio": dir_ratio,
            "overshoots": overshoots,
            "stopping_distance": stop_dist,
            "reaction_time": rt,
        })

    trials = pd.DataFrame(rows)
    if len(trials):
        if DROP_FIRST_MOVE:
            trials = trials[trials["trial_global"] != trials["trial_global"].min()].reset_index(drop=True)
        elif DROP_FIRST_PER_CONDITION:
            trials = trials[~trials["is_first_in_condition"]].reset_index(drop=True)
    return trials


# ======== EFFECTIVE THROUGHPUT (ISO 9241-9) ========

def effective_throughput_per_subject_model(trials):
    """
    Per (subject, model): effective throughput using means-of-means over
    conditions. We = 4.133 * SDx, where SDx is the std of selection-endpoint
    deviations projected onto the task axis, computed per condition. Ae is the
    mean actual amplitude in the condition.

    Note: dwell-based selection constrains endpoints to lie inside the target,
    which compresses SDx and inflates effective throughput in absolute terms.
    It remains valid for relative comparison across models under the same dwell.

    Also returns a penalized variant. We and the effective ID are still derived
    from successful endpoints (failed trials have no selection endpoint), but the
    movement-time denominator is taken over all trials in the condition (failed
    trials at their penalized time), so the metric is dragged down for models
    that fail often. A subject-model with no usable successful condition gets NaN
    for both, since effective width is undefined there.
    """
    out = []
    ok = trials[trials["success"] == 1].copy()
    for (subj, model), g in ok.groupby(["subject", "model"]):
        gall = trials[(trials["subject"] == subj) & (trials["model"] == model)]
        tp_conds = []
        tp_conds_pen = []
        for cond, gc in g.groupby("condition"):
            if len(gc) < MIN_TRIALS_FOR_WE:
                continue
            axis_dx = gc["target_x"].to_numpy() - gc["start_x"].to_numpy()
            axis_dy = gc["target_y"].to_numpy() - gc["start_y"].to_numpy()
            norm = np.hypot(axis_dx, axis_dy)
            norm[norm < 1e-9] = 1e-9
            ux, uy = axis_dx / norm, axis_dy / norm
            # Signed along-axis deviation of the endpoint from the target center.
            ex = gc["end_x"].to_numpy() - gc["target_x"].to_numpy()
            ey = gc["end_y"].to_numpy() - gc["target_y"].to_numpy()
            proj = ex * ux + ey * uy
            sdx = float(np.std(proj, ddof=1))
            if sdx < 1e-6:
                continue
            we = 4.133 * sdx
            ae = float(gc["distance"].mean())
            ide = math.log2(ae / we + 1.0) if we > 0 and ae > 0 else np.nan
            if np.isnan(ide):
                continue
            mt = float(gc["movement_time"].mean())
            if mt > 1e-9:
                tp_conds.append(ide / mt)
            # Penalized: same effective ID, MT over all trials in this condition.
            gc_all = gall[gall["condition"] == cond]
            mt_pen = float(gc_all["mt_penalized"].mean())
            if mt_pen > 1e-9:
                tp_conds_pen.append(ide / mt_pen)
        out.append({
            "subject": subj,
            "model": model,
            "throughput_effective": float(np.mean(tp_conds)) if tp_conds else np.nan,
            "throughput_effective_penalized": float(np.mean(tp_conds_pen)) if tp_conds_pen else np.nan,
        })
    return pd.DataFrame(out)


# ======== AGGREGATION ========

def aggregate_per_subject_model(trials):
    """Collapse trials to one row per (subject, model)."""
    ok = trials[trials["success"] == 1].copy()
    ok["tp_trial"] = ok["id_nominal"] / ok["movement_time"]

    # Per-condition means, for the means-of-means throughput and the regression.
    cond = (ok.groupby(["subject", "model", "condition"])
              .agg(id_cond=("id_nominal", "mean"),
                   mt_cond=("movement_time", "mean"))
              .reset_index())
    cond["tp_cond"] = cond["id_cond"] / cond["mt_cond"]
    mom = (cond.groupby(["subject", "model"])
               .agg(throughput_meanofmeans=("tp_cond", "mean"))
               .reset_index())

    agg = (ok.groupby(["subject", "model"])
             .agg(throughput_nominal=("tp_trial", "mean"),
                  movement_time=("movement_time", "mean"),
                  reaction_time=("reaction_time", "mean"),
                  overshoots=("overshoots", "mean"),
                  stopping_distance=("stopping_distance", "mean"),
                  n_success=("success", "size"))
             .reset_index())

    # Completion rate over all trials.
    comp = (trials.groupby(["subject", "model"])
                  .agg(n_trials=("success", "size"),
                       completion_rate=("success", "mean"))
                  .reset_index())
    comp["completion_rate"] = comp["completion_rate"] * 100.0

    # ======== all-trial metrics (failures folded in) ========
    allt = trials.copy()
    allt["tp_pen_trial"] = allt["id_nominal"] / allt["mt_penalized"]

    cond_pen = (allt.groupby(["subject", "model", "condition"])
                    .agg(id_cond=("id_nominal", "mean"),
                         mtp_cond=("mt_penalized", "mean"))
                    .reset_index())
    cond_pen["tp_cond_pen"] = cond_pen["id_cond"] / cond_pen["mtp_cond"]
    mom_pen = (cond_pen.groupby(["subject", "model"])
                       .agg(throughput_meanofmeans_penalized=("tp_cond_pen", "mean"))
                       .reset_index())

    agg_pen = (allt.groupby(["subject", "model"])
                   .agg(throughput_nominal_penalized=("tp_pen_trial", "mean"),
                        movement_time_penalized=("mt_penalized", "mean"),
                        path_efficiency=("path_efficiency", "mean"),
                        direction_change_ratio=("direction_change_ratio", "mean"))
                   .reset_index())

    eff = effective_throughput_per_subject_model(trials)

    psm = (agg.merge(mom, on=["subject", "model"], how="left")
              .merge(comp, on=["subject", "model"], how="left")
              .merge(agg_pen, on=["subject", "model"], how="left")
              .merge(mom_pen, on=["subject", "model"], how="left")
              .merge(eff, on=["subject", "model"], how="left"))
    return psm, cond


# ======== PER-CONDITION (PER-ID) AGGREGATION ========

def _effective_tp_for_condition(ok_cond, all_cond):
    """Effective throughput for a single condition. We and the effective ID come
    from the successful endpoints in that condition; returns (success-only,
    penalized) where the penalized variant divides the same effective ID by the
    mean penalized MT over all trials in the condition. NaN if too few successes
    or degenerate endpoint scatter (effective width is undefined there)."""
    if len(ok_cond) < MIN_TRIALS_FOR_WE:
        return np.nan, np.nan
    axis_dx = ok_cond["target_x"].to_numpy() - ok_cond["start_x"].to_numpy()
    axis_dy = ok_cond["target_y"].to_numpy() - ok_cond["start_y"].to_numpy()
    norm = np.hypot(axis_dx, axis_dy)
    norm[norm < 1e-9] = 1e-9
    ux, uy = axis_dx / norm, axis_dy / norm
    ex = ok_cond["end_x"].to_numpy() - ok_cond["target_x"].to_numpy()
    ey = ok_cond["end_y"].to_numpy() - ok_cond["target_y"].to_numpy()
    proj = ex * ux + ey * uy
    sdx = float(np.std(proj, ddof=1))
    if sdx < 1e-6:
        return np.nan, np.nan
    we = 4.133 * sdx
    ae = float(ok_cond["distance"].mean())
    ide = math.log2(ae / we + 1.0) if we > 0 and ae > 0 else np.nan
    if np.isnan(ide):
        return np.nan, np.nan
    mt = float(ok_cond["movement_time"].mean())
    mtp = float(all_cond["mt_penalized"].mean())
    eff = (ide / mt) if mt > 1e-9 else np.nan
    effp = (ide / mtp) if mtp > 1e-9 else np.nan
    return eff, effp


def aggregate_per_subject_model_condition(trials):
    """One row per (subject, model, condition). Each condition is a single
    nominal ID level, so this is the per-ID breakdown. Success-only metrics use
    the successful trials in the condition; penalized and completion use all
    trials in the condition. Same definitions as the global aggregation."""
    rows = []
    for (subj, model, condition), g in trials.groupby(["subject", "model", "condition"]):
        ok = g[g["success"] == 1]
        n_trials = len(g)
        n_success = len(ok)
        id_cond = float(g["id_nominal"].mean())
        mt_succ = float(ok["movement_time"].mean()) if n_success else np.nan
        mtp_all = float(g["mt_penalized"].mean())
        eff, effp = _effective_tp_for_condition(ok, g)
        rows.append({
            "subject": subj, "model": model, "condition": condition,
            "id_nominal": id_cond,
            "ring_radius": int(g["ring_radius"].iloc[0]),
            "width": float(g["width"].iloc[0]),
            "n_trials": n_trials, "n_success": n_success,
            "completion_rate": 100.0 * n_success / n_trials if n_trials else np.nan,
            "throughput_nominal": float((ok["id_nominal"] / ok["movement_time"]).mean()) if n_success else np.nan,
            "throughput_nominal_penalized": float((g["id_nominal"] / g["mt_penalized"]).mean()),
            "throughput_meanofmeans": (id_cond / mt_succ) if (mt_succ and not np.isnan(mt_succ)) else np.nan,
            "throughput_meanofmeans_penalized": (id_cond / mtp_all) if mtp_all > 1e-9 else np.nan,
            "throughput_effective": eff,
            "throughput_effective_penalized": effp,
            "movement_time": mt_succ,
            "movement_time_penalized": mtp_all,
            "path_efficiency": float(g["path_efficiency"].mean()),
            "direction_change_ratio": float(g["direction_change_ratio"].mean()),
            "overshoots": float(ok["overshoots"].mean()) if n_success else np.nan,
            "stopping_distance": float(ok["stopping_distance"].mean()) if n_success else np.nan,
            "reaction_time": float(ok["reaction_time"].mean()) if n_success else np.nan,
        })
    return pd.DataFrame(rows)


# ======== FITTS REGRESSION (VALIDITY CHECK) ========

def fitts_regression(cond):
    """
    Per model: grand regression of condition-mean MT on condition-mean ID
    (one point per condition, averaged across subjects), and the mean of
    per-subject regression R-squared values.
    """
    rows = []
    for model, g in cond.groupby("model"):
        grand = g.groupby("condition").agg(id=("id_cond", "mean"),
                                           mt=("mt_cond", "mean")).reset_index()
        if len(grand) >= 2 and grand["id"].nunique() >= 2:
            sl, ic, r, p, se = stats.linregress(grand["id"], grand["mt"])
            r2_grand = r ** 2
        else:
            sl = ic = r2_grand = np.nan

        subj_r2 = []
        for subj, gs in g.groupby("subject"):
            gg = gs.groupby("condition").agg(id=("id_cond", "mean"),
                                             mt=("mt_cond", "mean")).reset_index()
            if len(gg) >= 2 and gg["id"].nunique() >= 2:
                _, _, rr, _, _ = stats.linregress(gg["id"], gg["mt"])
                subj_r2.append(rr ** 2)
        rows.append({
            "model": model,
            "slope": sl,
            "intercept": ic,
            "r2_grand": r2_grand,
            "r2_persubject_mean": float(np.mean(subj_r2)) if subj_r2 else np.nan,
            "n_conditions": int(grand["condition"].nunique()),
            "n_subjects": int(g["subject"].nunique()),
        })
    return pd.DataFrame(rows)


# ======== STATISTICS ========

def _holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def _cohens_dz(diff):
    diff = diff[~np.isnan(diff)]
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 1e-12 else np.nan


def _rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank."""
    d = a - b
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total > 0 else np.nan


def stats_for_metric(psm, metric):
    """Friedman omnibus and Holm-corrected pairwise Wilcoxon for one metric."""
    wide = psm.pivot_table(index="subject", columns="model", values=metric)
    models = list(wide.columns)

    complete = wide.dropna(axis=0, how="any")
    friedman = {"metric": metric, "n_complete_subjects": int(len(complete)),
                "n_models": len(models), "chi2": np.nan, "p": np.nan,
                "kendall_w": np.nan}
    if len(complete) >= 2 and len(models) >= 3:
        try:
            chi2, p = stats.friedmanchisquare(*[complete[m].to_numpy() for m in models])
            friedman.update(chi2=float(chi2), p=float(p),
                            kendall_w=float(chi2 / (len(complete) * (len(models) - 1))))
        except Exception:
            pass

    pairs = []
    raw_p = []
    for a, b in combinations(models, 2):
        sub = wide[[a, b]].dropna(axis=0, how="any")
        if len(sub) < 2:
            continue
        va, vb = sub[a].to_numpy(), sub[b].to_numpy()
        diff = va - vb
        try:
            w, p = stats.wilcoxon(va, vb, zero_method="wilcox", correction=False,
                                  mode="auto")
        except Exception:
            w, p = np.nan, np.nan
        pairs.append({
            "metric": metric, "model_a": a, "model_b": b, "n_pairs": int(len(sub)),
            "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "wilcoxon_W": float(w) if not np.isnan(w) else np.nan,
            "p_raw": float(p) if not np.isnan(p) else np.nan,
            "cohens_dz": _cohens_dz(diff),
            "rank_biserial": _rank_biserial(va, vb),
        })
        raw_p.append(p)

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        valid = pairs_df["p_raw"].notna()
        pairs_df.loc[valid, "p_holm"] = _holm_bonferroni(pairs_df.loc[valid, "p_raw"].to_numpy())
        pairs_df["significant_holm_0.05"] = pairs_df["p_holm"] < 0.05
    return friedman, pairs_df


# ======== SUMMARY ========

def summary_by_model(psm, metrics):
    rows = []
    for model, g in psm.groupby("model"):
        for met in metrics:
            v = g[met].dropna().to_numpy()
            if v.size == 0:
                continue
            rows.append({
                "model": model, "metric": met, "n_subjects": int(v.size),
                "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
                "median": float(np.median(v)),
                "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                "p25": float(np.percentile(v, 25)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "higher_is_better": HIGHER_IS_BETTER.get(met, True),
            })
    summ = pd.DataFrame(rows)
    # Rank by mean (respecting direction) and by consistency (lower std better).
    ranked = []
    for met, g in summ.groupby("metric"):
        g = g.copy()
        asc = not HIGHER_IS_BETTER.get(met, True)
        g["rank_mean"] = g["mean"].rank(ascending=asc, method="min")
        g["rank_consistency_std"] = g["std"].rank(ascending=True, method="min")
        ranked.append(g)
    return pd.concat(ranked, ignore_index=True) if ranked else summ


# ======== PLOTS ========

def _order_models(psm, metric=None):
    if not FIX_SORT and metric:
        means = psm.groupby("model")[metric].mean()
        return list(means.sort_values(ascending=True).index)

    present = list(psm["model"].unique())
    ordered = [m for m in MODEL_ORDER if m in present]
    extra = sorted(m for m in present if m not in MODEL_ORDER)
    return ordered + extra


def _model_color_map(models):
    # Deterministic color per canonical model, keyed by MODEL_ORDER position so a
    # model keeps the same color across every panel and every ID level. Models
    # not in MODEL_ORDER fall after the listed ones.
    present = set(models)
    ordered = [m for m in MODEL_ORDER if m in present] + \
              sorted(m for m in present if m not in MODEL_ORDER)
    cols = plt.cm.tab10.colors
    return {m: cols[i % len(cols)] for i, m in enumerate(ordered)}


def _fmt_stat(v):
    if v is None or not np.isfinite(v):
        return "nan"
    return ("%.1f" % v) if abs(v) >= 10 else ("%.2f" % v)


def plot_metric_box(psm, metric, ax, color_map=None):
    data = psm[["model", metric]].dropna()
    if data.empty:
        ax.set_visible(False)
        return
    order = _order_models(psm, metric)
    if color_map is None:
        color_map = _model_color_map(psm["model"].unique())
    colors = [color_map.get(model, "gray") for model in order]
    if _HAS_SNS:
        sns.boxplot(data=data, x="model", y=metric, order=order, ax=ax,
                    showfliers=False, width=0.6,
                    whiskerprops=dict(alpha=0.7),
                    medianprops=dict(color="black", linewidth=0.6))
        # Per-box fill, kept translucent so the jittered points stay visible.
        for i, patch in enumerate(ax.patches):
            r, g, b = colors[i % len(colors)][:3]
            patch.set_facecolor((r, g, b, 0.45))
            patch.set_edgecolor((r, g, b, 0.9))
        sns.stripplot(data=data, x="model", y=metric, order=order, ax=ax,
                      color="black", alpha=0.55, size=4, jitter=0.18)
    else:
        groups = [data[data["model"] == m][metric].to_numpy() for m in order]
        ax.boxplot(groups, labels=[DISPLAY_NAMES.get(m, m) for m in order], showfliers=False)

    # Mean and std written above each box, clear of the jittered points.
    grp = data.groupby("model")[metric]
    means = grp.mean()
    stds = grp.std(ddof=1)
    vmin = float(data[metric].min())
    vmax = float(data[metric].max())
    vr = (vmax - vmin) or 1.0
    for i, m in enumerate(order):
        col_max = float(data[data["model"] == m][metric].max())
        sd = stds[m] if (m in stds.index and np.isfinite(stds[m])) else 0.0
        ax.text(i, col_max + 0.03 * vr,
                "mean %s\nstd %s" % (_fmt_stat(means[m]), _fmt_stat(sd)),
                ha="center", va="bottom", fontsize=6.5, color="black",
                linespacing=0.95)
    ax.set_ylim(vmin - 0.05 * vr, vmax + 0.22 * vr)

    direction = "higher better" if HIGHER_IS_BETTER.get(metric, True) else "lower better"
    ax.set_title("%s (%s)" % (metric, direction), fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Short display labels, fixed order.
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in order])
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)


def plot_boxgrid(psm, metrics, path):
    metrics = [m for m in metrics if m in psm.columns and psm[m].notna().any()]
    n = len(metrics)
    ncol = 4
    nrow = int(math.ceil(n / ncol))
    cmap = _model_color_map(psm["model"].unique())
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), dpi=400)
    axes = np.atleast_1d(axes).flatten()
    for i, met in enumerate(metrics):
        plot_metric_box(psm, met, axes[i], color_map=cmap)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Online Fitts metrics across models (each point is one subject)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_individual_boxes(psm, metrics, out_dir):
    cmap = _model_color_map(psm["model"].unique())
    for met in metrics:
        if met not in psm.columns or not psm[met].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * psm["model"].nunique()), 5), dpi=200)
        plot_metric_box(psm, met, ax, color_map=cmap)
        ax.set_ylabel(met)
        fig.tight_layout()
        fig.savefig(join(out_dir, "%s.png" % met), bbox_inches="tight")
        plt.close(fig)


def plot_metric_by_id_box(psmc, metric, path, color_map=None):
    # One figure per metric, one boxplot subplot per ID level (same box style as
    # the headline plots). Each panel shows the across-subject distribution per
    # model at that single ID. A fixed color map keeps a model's color identical
    # across panels even when a sparse metric drops a model in some panel.
    if metric not in psmc.columns or not psmc[metric].notna().any():
        return
    if color_map is None:
        color_map = _model_color_map(psmc["model"].unique())
    canon = psmc.groupby("condition")["id_nominal"].mean().sort_values()
    conds = list(canon.index)
    n = len(conds)
    ncol = 2 if not (n % 4) else 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), dpi=300,
                             squeeze=False)
    axes = axes.flatten()
    direction = "higher better" if HIGHER_IS_BETTER.get(metric, True) else "lower better"
    for i, c in enumerate(conds):
        sub = psmc[psmc["condition"] == c][["model", metric]].dropna()
        plot_metric_box(sub, metric, axes[i], color_map=color_map)
        axes[i].set_title("ID=%.2f  (%s)" % (float(canon[c]), c), fontsize=9)
        axes[i].set_ylabel(metric, fontsize=8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("%s by ID (%s) -- each point is one subject" % (metric, direction),
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all_by_id(psmc, metrics, out_dir, color_map=None):
    if color_map is None:
        color_map = _model_color_map(psmc["model"].unique())
    for met in metrics:
        if met not in psmc.columns or not psmc[met].notna().any():
            continue
        plot_metric_by_id_box(psmc, met, join(out_dir, "%s_by_id.png" % met), color_map)


def plot_regression(cond, reg, path):
    models = _order_models(cond)
    ncol = min(4, len(models))
    nrow = int(math.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), dpi=200,
                             squeeze=False)
    axes = axes.flatten()
    rmap = reg.set_index("model")
    for i, model in enumerate(models):
        ax = axes[i]
        g = cond[cond["model"] == model]
        grand = g.groupby("condition").agg(id=("id_cond", "mean"),
                                           mt=("mt_cond", "mean")).reset_index()
        ax.scatter(g["id_cond"], g["mt_cond"], s=12, alpha=0.3, color="gray",
                   label="subject x condition")
        ax.scatter(grand["id"], grand["mt"], s=45, color="C0",
                   label="condition mean", zorder=5)
        if model in rmap.index and not np.isnan(rmap.loc[model, "slope"]):
            sl, ic = rmap.loc[model, "slope"], rmap.loc[model, "intercept"]
            xs = np.linspace(g["id_cond"].min(), g["id_cond"].max(), 50)
            ax.plot(xs, ic + sl * xs, color="C3", linewidth=2)
            ax.set_title("%s\nR2(grand)=%.3f  R2(subj)=%.3f" % (
                DISPLAY_NAMES.get(model, model), rmap.loc[model, "r2_grand"],
                rmap.loc[model, "r2_persubject_mean"]), fontsize=8)
        else:
            ax.set_title(DISPLAY_NAMES.get(model, model), fontsize=8)
        ax.set_xlabel("ID (bits)", fontsize=8)
        ax.set_ylabel("MT (s)", fontsize=8)
        ax.grid(alpha=0.25)
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Fitts validity: movement time vs index of difficulty", fontsize=12,
                 y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ======== MAIN ========

def main(root=FITTS_ROOT, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    logs = discover_logs(root)
    if not logs:
        raise RuntimeError("No usable logs found under %s" % root)

    all_trials = []
    manifest = []
    for subj, fname, path in logs:
        df = load_log(path)
        model = df["model"].iloc[0] if "model" in df.columns and pd.notna(df["model"].iloc[0]) \
            else _model_from_filename(fname)
        model = str(model).strip()
        trials = segment_trials(df, subject=str(subj), model=model)
        all_trials.append(trials)
        manifest.append({"subject": str(subj), "model": model, "file": fname,
                         "n_trials": len(trials)})

    trials = pd.concat(all_trials, ignore_index=True)
    pd.DataFrame(manifest).to_csv(join(out_dir, "manifest.csv"), index=False)
    trials.to_csv(join(out_dir, "trials_long.csv"), index=False)

    psm, cond = aggregate_per_subject_model(trials)
    psm.to_csv(join(out_dir, "per_subject_model.csv"), index=False)
    cond.to_csv(join(out_dir, "per_subject_model_condition.csv"), index=False)

    psmc = aggregate_per_subject_model_condition(trials)
    psmc.to_csv(join(out_dir, "per_subject_model_condition_metrics.csv"), index=False)

    reg = fitts_regression(cond)
    reg.to_csv(join(out_dir, "fitts_regression.csv"), index=False)

    metrics_present = [m for m in PLOT_METRICS if m in psm.columns and psm[m].notna().any()]
    summ = summary_by_model(psm, metrics_present)
    summ.to_csv(join(out_dir, "summary_by_model.csv"), index=False)

    friedman_rows = []
    all_pairs = []
    for met in metrics_present:
        fr, pairs = stats_for_metric(psm, met)
        friedman_rows.append(fr)
        if len(pairs):
            all_pairs.append(pairs)
    pd.DataFrame(friedman_rows).to_csv(join(out_dir, "friedman_omnibus.csv"), index=False)
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            join(out_dir, "pairwise_wilcoxon.csv"), index=False)

    plot_boxgrid(psm, metrics_present, join(out_dir, "metrics_grid.png"))
    plot_individual_boxes(psm, metrics_present, out_dir)

    byid_metrics = [m for m in PLOT_METRICS if m in psmc.columns and psmc[m].notna().any()]
    cmap = _model_color_map(psm["model"].unique())
    plot_all_by_id(psmc, byid_metrics, out_dir, cmap)
    if len(cond):
        plot_regression(cond, reg, join(out_dir, "fitts_regression.png"))

    n_subj = psm["subject"].nunique()
    n_models = psm["model"].nunique()
    print("Subjects: %d   Models: %d   Trials: %d" % (n_subj, n_models, len(trials)))
    print("Outputs written to: %s" % out_dir)
    return {"trials": trials, "per_subject_model": psm, "condition": cond,
            "condition_metrics": psmc, "regression": reg, "summary": summ}


if __name__ == "__main__":
    main()