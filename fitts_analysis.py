"""
Fitts' law analysis for online (real-time) myoelectric model evaluation.

Reads per-subject, per-model ISO 9241-9 (Mode B, circular ring) cursor logs and
computes the standard online myoelectric control metrics (throughput, path
efficiency, completion rate, overshoot, reaction time, stopping distance),
validates the Fitts task via the movement-time vs index-of-difficulty
regression, runs repeated-measures statistics across models (Friedman, pairwise
Wilcoxon with Holm-Bonferroni correction, Cohen's dz, matched-pairs
rank-biserial), and renders per-metric boxplots with per-subject jitter.

Metric definitions follow Scheme and Englehart 2013 (IEEE TNSRE), Wurth and
Hargrove 2014 (J NeuroEng Rehabil), Eddy et al. 2023 (LibEMG, IEEE Access), and
Waris et al. 2018 / 2020. Effective throughput follows the ISO 9241-9
effective-width method (Soukoreff and MacKenzie 2004; We = 4.133 * SDx).

The unit of analysis is the subject. Higher mean with lower across-subject
spread is treated as the target outcome.
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

# Task parameters, must match the run that produced the logs.
FRAME_RATE = 60
HOLD_FRAMES = 30
TIMEOUT_FRAMES = 420
SCREEN_SIZE = (1690, 980)
RING_RADII = [300, 450]
TARGET_RADII = [20, 10]

# Screen center. None infers it per file from the first logged cursor position
# (the cursor always starts at screen center). Fallback is SCREEN_SIZE / 2.
SCREEN_CENTER = None

# Drop the first trial of each condition (the centering or cross-condition
# transition move), per standard ISO multidirectional practice. Per-trial
# nominal throughput uses the actual logged amplitude either way.
DROP_FIRST_PER_CONDITION = True

# Quality metrics (throughput, MT, PE, RT, overshoot, stopping distance) are
# computed over successfully acquired trials only. Completion rate uses all
# (non-dropped) trials.
MOVE_STEP_PX = 1.0          # cursor step magnitude that counts as movement onset
DWELL_TIME = HOLD_FRAMES / FRAME_RATE
MIN_TRIALS_FOR_WE = 3       # minimum successful trials per condition for We

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
}

# Metrics shown in the headline boxplot grid.
PLOT_METRICS = [
    "throughput_nominal",
    "path_efficiency",
    "completion_rate",
    "overshoots",
    "movement_time",
    "stopping_distance",
    "reaction_time",
    "throughput_effective",
]


# ======== LOG DISCOVERY AND LOADING ========

COLUMNS = ["time", "frame", "mode", "model", "cursor_x", "cursor_y",
           "target_x", "target_y", "radius", "X", "Y", "vx", "vy",
           "acc_x", "acc_y", "inside", "hold_count", "velocity",
           "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]

NUM_COLS = ["time", "frame", "cursor_x", "cursor_y", "target_x", "target_y",
            "radius", "X", "Y", "vx", "vy", "acc_x", "acc_y", "inside",
            "hold_count", "velocity"]


def _is_test_name(name):
    return name.lower().startswith("test")


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


def segment_trials(df, subject, model):
    """
    Split one session into trials and compute per-trial metrics.

    A trial is a maximal run of rows sharing the same (target_x, target_y).
    Acquisition (hold_count reaching HOLD_FRAMES) is logged on the FIRST row of
    the NEXT trial because of the known off-by-one in the runner; the acquired
    target therefore lives on the current trial's rows and the selection point
    is the current trial's last logged cursor position.
    """
    cx, cy = _infer_center(df)

    tx = df["target_x"].to_numpy()
    ty = df["target_y"].to_numpy()
    changed = (np.diff(tx) != 0) | (np.diff(ty) != 0)
    block_id = np.concatenate([[0], np.cumsum(changed)])
    df = df.assign(_block=block_id)

    blocks = [g for _, g in df.groupby("_block", sort=True)]
    rows = []

    # Within-condition running index, to flag the first trial of each
    # contiguous condition run.
    prev_cond = None
    cond_run_idx = -1

    for bi, blk in enumerate(blocks):
        blk = blk.reset_index(drop=True)
        n = len(blk)
        t_x = float(blk["target_x"].iloc[0])
        t_y = float(blk["target_y"].iloc[0])
        radius = float(blk["radius"].iloc[0])
        width = 2.0 * radius

        start = (float(blk["cursor_x"].iloc[0]), float(blk["cursor_y"].iloc[0]))
        end = (float(blk["cursor_x"].iloc[-1]), float(blk["cursor_y"].iloc[-1]))

        amplitude = math.hypot(start[0] - t_x, start[1] - t_y)
        ring_raw = math.hypot(t_x - cx, t_y - cy)
        ring = min(RING_RADII, key=lambda r: abs(r - ring_raw)) if RING_RADII else round(ring_raw)
        condition = "ring%d_w%d" % (int(ring), int(width))

        if condition != prev_cond:
            cond_run_idx += 1
            is_first_in_cond = True
        else:
            is_first_in_cond = False
        prev_cond = condition

        # Outcome. Acquisition (hold_count reaching HOLD_FRAMES) is logged on the
        # NEXT block's first row because of the runner off-by-one, so a real
        # acquisition is read from the next block, not from this block's rows. A
        # leading hold_count == HOLD_FRAMES on this block's own first row is the
        # PREVIOUS target's acquisition artifact: the target has already switched,
        # the cursor is still on the old target, and inside == 0. Genuine dwell
        # therefore only accumulates while inside == 1, so it is gated on that to
        # keep the artifact from inflating the in-block hold maximum. In Mode B
        # (ISO ring) consecutive targets are far apart, so the artifact row is
        # always inside == 0.
        inside_arr = blk["inside"].to_numpy().astype(int)
        hold_arr = blk["hold_count"].to_numpy()
        genuine_hold = hold_arr[inside_arr == 1]
        genuine_max_hold = float(genuine_hold.max()) if genuine_hold.size else 0.0
        if bi + 1 < len(blocks):
            nxt = blocks[bi + 1].reset_index(drop=True)
            next_first_hold = float(nxt["hold_count"].iloc[0])
            acquired = next_first_hold >= HOLD_FRAMES
            t_acq = float(nxt["time"].iloc[0])
        else:
            # Final trial of the file: the acquisition row is not written
            # because the session closes. Accept a near-complete genuine dwell.
            acquired = genuine_max_hold >= (HOLD_FRAMES - 1)
            t_acq = float(blk["time"].iloc[-1]) + 1.0 / FRAME_RATE

        timed_out = (not acquired) and (n >= TIMEOUT_FRAMES * 0.9)
        success = bool(acquired)

        t_start = float(blk["time"].iloc[0])
        mt = (t_acq - t_start) if success else np.nan

        # Path length over the block.
        dx = np.diff(blk["cursor_x"].to_numpy())
        dy = np.diff(blk["cursor_y"].to_numpy())
        path_len = float(np.hypot(dx, dy).sum())
        pe = (amplitude / path_len) if path_len > 1e-9 else np.nan
        if pe is not None and not np.isnan(pe):
            pe = min(pe, 1.0)

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

        id_nominal = math.log2(amplitude / width + 1.0) if width > 0 and amplitude > 0 else np.nan
        endpoint_dev = None  # filled later, needs task axis (handled in effective TP)

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
            "amplitude": amplitude,
            "id_nominal": id_nominal,
            "start_x": start[0], "start_y": start[1],
            "target_x": t_x, "target_y": t_y,
            "end_x": end[0], "end_y": end[1],
            "success": int(success),
            "timed_out": int(timed_out),
            "movement_time": mt,
            "movement_time_no_dwell": (mt - DWELL_TIME) if (success and not np.isnan(mt)) else np.nan,
            "path_length": path_len,
            "path_efficiency": (pe * 100.0) if pe is not None and not np.isnan(pe) else np.nan,
            "overshoots": overshoots,
            "stopping_distance": stop_dist,
            "reaction_time": rt,
        })

    trials = pd.DataFrame(rows)
    if DROP_FIRST_PER_CONDITION and len(trials):
        trials = trials[~trials["is_first_in_condition"]].reset_index(drop=True)
    return trials


# ======== EFFECTIVE THROUGHPUT (ISO 9241-9) ========

def effective_throughput_per_subject_model(trials):
    """
    Per (subject, model): effective throughput using means-of-means over
    conditions. We = 4.133 * SDx, where SDx is the std of selection-endpoint
    deviations projected onto the task axis. Ae is the mean actual amplitude.

    Note: dwell-based selection constrains endpoints to lie inside the target,
    which compresses SDx and inflates effective throughput in absolute terms.
    It remains valid for relative comparison across models under the same dwell.
    """
    out = []
    ok = trials[trials["success"] == 1].copy()
    for (subj, model), g in ok.groupby(["subject", "model"]):
        tp_conds = []
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
            ae = float(gc["amplitude"].mean())
            ide = math.log2(ae / we + 1.0) if we > 0 and ae > 0 else np.nan
            mt = float(gc["movement_time"].mean())
            if mt > 1e-9 and not np.isnan(ide):
                tp_conds.append(ide / mt)
        out.append({
            "subject": subj,
            "model": model,
            "throughput_effective": float(np.mean(tp_conds)) if tp_conds else np.nan,
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
                  movement_time_no_dwell=("movement_time_no_dwell", "mean"),
                  path_efficiency=("path_efficiency", "mean"),
                  reaction_time=("reaction_time", "mean"),
                  overshoots=("overshoots", "mean"),
                  stopping_distance=("stopping_distance", "mean"),
                  n_success=("success", "size"))
             .reset_index())

    # Completion rate over all (non-dropped) trials.
    comp = (trials.groupby(["subject", "model"])
                  .agg(n_trials=("success", "size"),
                       completion_rate=("success", "mean"))
                  .reset_index())
    comp["completion_rate"] = comp["completion_rate"] * 100.0

    eff = effective_throughput_per_subject_model(trials)

    psm = (agg.merge(mom, on=["subject", "model"], how="left")
              .merge(comp, on=["subject", "model"], how="left")
              .merge(eff, on=["subject", "model"], how="left"))
    return psm, cond


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

def _order_models(psm, metric):
    med = psm.groupby("model")[metric].median()
    asc = not HIGHER_IS_BETTER.get(metric, True)
    return list(med.sort_values(ascending=asc).index)


def plot_metric_box(psm, metric, ax):
    data = psm[["model", metric]].dropna()
    if data.empty:
        ax.set_visible(False)
        return
    order = _order_models(psm, metric)
    if _HAS_SNS:
        sns.boxplot(data=data, x="model", y=metric, order=order, ax=ax,
                    showfliers=False, width=0.6,
                    boxprops=dict(alpha=0.45), whiskerprops=dict(alpha=0.7),
                    medianprops=dict(color="black", linewidth=2))
        sns.stripplot(data=data, x="model", y=metric, order=order, ax=ax,
                      color="black", alpha=0.55, size=4, jitter=0.18)
    else:
        groups = [data[data["model"] == m][metric].to_numpy() for m in order]
        ax.boxplot(groups, labels=order, showfliers=False)
    direction = "higher better" if HIGHER_IS_BETTER.get(metric, True) else "lower better"
    ax.set_title("%s (%s)" % (metric, direction), fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)


def plot_boxgrid(psm, metrics, path):
    metrics = [m for m in metrics if m in psm.columns and psm[m].notna().any()]
    n = len(metrics)
    ncol = 4
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), dpi=400)
    axes = np.atleast_1d(axes).flatten()
    for i, met in enumerate(metrics):
        plot_metric_box(psm, met, axes[i])
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Online Fitts metrics across models (each point is one subject)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_individual_boxes(psm, metrics, out_dir):
    for met in metrics:
        if met not in psm.columns or not psm[met].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * psm["model"].nunique()), 5), dpi=200)
        plot_metric_box(psm, met, ax)
        ax.set_ylabel(met)
        fig.tight_layout()
        fig.savefig(join(out_dir, "box_%s.png" % met), bbox_inches="tight")
        plt.close(fig)


def plot_regression(cond, reg, path):
    models = list(cond["model"].unique())
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
                model, rmap.loc[model, "r2_grand"], rmap.loc[model, "r2_persubject_mean"]),
                fontsize=8)
        else:
            ax.set_title(model, fontsize=8)
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

    plot_boxgrid(psm, metrics_present, join(out_dir, "boxplots_grid.png"))
    plot_individual_boxes(psm, metrics_present, out_dir)
    if len(cond):
        plot_regression(cond, reg, join(out_dir, "fitts_regression.png"))

    n_subj = psm["subject"].nunique()
    n_models = psm["model"].nunique()
    print("Subjects: %d   Models: %d   Trials: %d" % (n_subj, n_models, len(trials)))
    print("Outputs written to: %s" % out_dir)
    return {"trials": trials, "per_subject_model": psm, "condition": cond,
            "regression": reg, "summary": summ}


if __name__ == "__main__":
    main()
