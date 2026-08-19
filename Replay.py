import os
import re
import math
import sys
from os.path import join, isdir, exists

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSlider, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap, QFont, QLinearGradient,
    QKeySequence, QShortcut)


# ======== SESSION SELECTION ========

USER_ID = "3"
# MODEL = "within_cnnhcf_raw_base-5" 
MODEL = "cross_mhcnn_segmented_base"
FITTS_ROOT = "fitts_logs"

LOG_PATH = None

# Which session to take when a subject has several logs for the same model.
# -1 is the most recent by filename timestamp, 0 the earliest.
SESSION_INDEX = -1


# ======== TRACE CONFIG ========

TRACE_ENABLED = True

TRACE_SCOPE = 'condition'
TRACE_WINDOW_FRAMES = 240

TRACE_KEEP_PAST = False

# 'speed', 'gesture', 'time' or 'none' (single flat colour).
TRACE_COLOR_BY = 'speed'
TRACE_CMAP = 'stoplight'
TRACE_WIDTH = 2.6
TRACE_MARK_TRIALS = True

# Upper end of the speed colour ramp, px/s. None auto-fits to the 98th
# percentile of the session so a single outlier frame cannot flatten the map.
# Fix this manually when producing figures that must be compared across models.
SPEED_VMAX = None

SHOW_COLORBAR = True
SHOW_ANNOTATION = True


# ======== DISPLAY CONFIG ========

PAPER_MODE = True

# Shrinks the replay window when the task resolution exceeds the display.
# All drawing is scaled, logged coordinates are untouched.
VIEW_SCALE = 0.85

INITIAL_SPEED = 1.0
START_PAUSED = False
LOOP = False

SHOT_DIR = "replay_shots"
SHOT_SCALE = 2.0


# ======== OVERLAY CONFIG ========

# Sizes below are given in task pixels and are multiplied at draw time by the
# render scale (window width / SCREEN_SIZE width). That is 0.85 in the live
# window and SHOT_SCALE in an export, so the overlay keeps the same relative
# size in both instead of shrinking in the saved figure.


COLORBAR_ORIENT = 'vertical'
COLORBAR_LENGTH_FRAC = 0.62
COLORBAR_THICKNESS = 46
COLORBAR_TICKS = 6
COLORBAR_TITLE_PT = 17
COLORBAR_FONT_PT = 15
COLORBAR_MARGIN = 34

# Swatch edge length for the gesture legend.
GESTURE_SWATCH = 28

ANNOTATION_TITLE_PT = 21
ANNOTATION_BODY_PT = 16
ANNOTATION_MARGIN = 18

# Translucent box behind the overlays. Useful over the dark live theme, only
# clutter on a white paper background.
OVERLAY_PANEL = False

DISPLAY_NAMES = {
    "within_mhcnn_raw_base-ft-5": "FT-5",
    "within_mhcnn_raw_base-ft-1": "FT-1",
    "within_cnnhcf_raw_base-5": "Within-5",
    "cross_mhcnn_segmented_base": "Segmented",
    "cross_mhcnn_raw_base-rn": "RunNorm",
    "cross_mhcnn_raw_base": "Base",
    "cross_mhcnn_raw_1va": "Proto",
    "cross_mhcnn_raw_rest": "RestLoss",
    "cross_mhcnn_raw_trp": "Triplet",
}

GESTURE_NAMES = {0: "NM", 1: "HC", 2: "FX", 3: "EX", 4: "HO"}


# ======== RUNNER CONSTANTS ========

try:
    from utils import PARAMS
    FRAME_RATE = PARAMS['frame_rate']
    HOLD_FRAMES = PARAMS['hold_frames_required']
    TIMEOUT_FRAMES = PARAMS['target_timeout_frames']
    SCREEN_SIZE = PARAMS['screen_size']
except Exception:
    FRAME_RATE = 60
    HOLD_FRAMES = 45
    TIMEOUT_FRAMES = 480
    SCREEN_SIZE = (1690, 980)

# Quantization of the target distance when grouping trials into conditions,
# in pixels. Only needs to be smaller than the gap between configured ring
# radii and larger than the integer rounding of the ring point coordinates.
COND_D_TOL = 25.0

COLUMNS = ["time", "frame", "mode", "model", "cursor_x", "cursor_y",
           "target_x", "target_y", "radius", "X", "Y", "vx", "vy",
           "acc_x", "acc_y", "inside", "hold_count", "velocity",
           "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]

NUM_COLS = ["time", "frame", "cursor_x", "cursor_y", "target_x", "target_y",
            "radius", "X", "Y", "vx", "vy", "acc_x", "acc_y", "inside",
            "hold_count", "velocity", "probs_0", "probs_1", "probs_2",
            "probs_3", "probs_4"]


# ======== THEME ========

THEME_DARK = {
    'bg': QColor("#252525"),
    'ring': QColor(0, 0, 0, 200),
    'target_idle': QColor("#BD1B1B"),
    'target_hold': QColor("#10DA39"),
    'cursor': QColor("#10C2DA"),
    'cursor_edge': QColor(255, 255, 255, 200),
    'ghost': QColor(255, 255, 255, 45),
    'text': QColor(235, 235, 235),
    'flat_trace': QColor("#10C2DA"),
    'panel': QColor(0, 0, 0, 120),
}

THEME_PAPER = {
    'bg': QColor("#FFFFFF"),
    'ring': QColor(150, 150, 150),
    'target_idle': QColor(200, 60, 60),
    'target_hold': QColor(40, 160, 70),
    'cursor': QColor(20, 20, 20),
    'cursor_edge': QColor(255, 255, 255),
    'ghost': QColor(0, 0, 0, 40),
    'text': QColor(30, 30, 30),
    'flat_trace': QColor(20, 20, 20),
    'panel': QColor(255, 255, 255, 200),
}


# ======== COLOUR MAPS ========

CMAPS = {
    'viridis': [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121),
                (109, 199, 82), (180, 222, 44), (253, 231, 37)],
    'magma': [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
              (181, 54, 122), (229, 80, 100), (251, 135, 97),
              (254, 194, 135), (252, 253, 191)],
    'turbo': [(48, 18, 59), (70, 107, 227), (36, 180, 223), (91, 229, 131),
              (176, 245, 60), (250, 205, 44), (250, 120, 32),
              (204, 39, 10), (122, 4, 3)],

    'stoplight': [(74, 12, 20), (114, 22, 24), (150, 34, 22), (170, 54, 20),
              (184, 72, 20), (196, 92, 22), (204, 114, 24),
              (200, 140, 28), (156, 178, 34), (90, 215, 70)],
}

# Okabe-Ito, colour-blind safe. Rest is neutral grey so the active classes
# carry all the visual weight.
GESTURE_COLORS = {
    0: QColor(150, 150, 150),
    1: QColor(0, 114, 178),
    2: QColor(230, 159, 0),
    3: QColor(0, 158, 115),
    4: QColor(204, 121, 167),
}


def build_lut(name, n=256):
    """Discretize a control-point colour map into n QColor steps."""
    pts = CMAPS.get(name, CMAPS['viridis'])
    arr = np.array(pts, dtype=float)
    k = len(arr) - 1
    lut = []
    for i in range(n):
        t = i / (n - 1) * k
        lo = int(math.floor(t))
        hi = min(lo + 1, k)
        f = t - lo
        rgb = arr[lo] * (1.0 - f) + arr[hi] * f
        lut.append(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return lut


# ======== LOG DISCOVERY AND LOADING ========

def _model_from_filename(fname):
    stem = re.sub(r"\.csv$", "", fname, flags=re.IGNORECASE)
    m = re.match(r"^Fitts_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$", stem)
    return m.group(1) if m else stem


def find_log(root, user, model, index=-1):
    folder = join(root, str(user))
    if not isdir(folder):
        raise FileNotFoundError("No log folder for subject %s at %s" % (user, folder))
    found = []
    tags = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        if 'test' in fname.lower():
            continue
        tag = _model_from_filename(fname)
        tags.append(tag)
        if tag == model or tag.startswith(model + "_"):
            found.append(join(folder, fname))
    if not found:
        raise FileNotFoundError(
            "No log for model '%s' under %s. Available: %s"
            % (model, folder, ", ".join(sorted(set(tags))) or "none"))
    return found[index]


def load_log(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "cursor_x", "cursor_y", "target_x", "target_y"])
    df = df.sort_values("frame").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Log has fewer than two usable frames: %s" % filepath)
    return df


# ======== SESSION MODEL ========

class Session(object):
    """Arrays and per-trial structure for one log, precomputed once."""

    def __init__(self, df, path):
        self.path = path
        self.df = df
        self.n = len(df)

        self.t = df["time"].to_numpy(dtype=float)
        self.t = self.t - self.t[0]
        # Guard against a non-monotonic clock so seeking stays well defined.
        self.t = np.maximum.accumulate(self.t)

        self.cx = df["cursor_x"].to_numpy(dtype=float)
        self.cy = df["cursor_y"].to_numpy(dtype=float)
        self.tx = df["target_x"].to_numpy(dtype=float)
        self.ty = df["target_y"].to_numpy(dtype=float)
        self.rad = df["radius"].to_numpy(dtype=float)
        self.hold = df["hold_count"].to_numpy(dtype=float)
        self.inside = df["inside"].to_numpy(dtype=float)

        self.mode = str(df["mode"].iloc[0]) if "mode" in df.columns else "B"
        self.model = str(df["model"].iloc[0]) if "model" in df.columns \
            else _model_from_filename(os.path.basename(path))

        prob_cols = [c for c in ["probs_0", "probs_1", "probs_2", "probs_3", "probs_4"]
                     if c in df.columns]
        if prob_cols:
            probs = df[prob_cols].to_numpy(dtype=float)
            probs = np.nan_to_num(probs, nan=-1e9)
            self.gesture = probs.argmax(axis=1)
            self.probs = probs
        else:
            self.gesture = np.zeros(self.n, dtype=int)
            self.probs = np.zeros((self.n, 5))

        # Per-segment speed in px/s from the drawn path itself.
        dx = np.diff(self.cx)
        dy = np.diff(self.cy)
        dt = np.diff(self.t)
        dt = np.where(dt <= 1e-9, 1.0 / FRAME_RATE, dt)
        self.seg_speed = np.hypot(dx, dy) / dt
        moving = self.seg_speed[self.seg_speed > 1e-6]
        auto_vmax = float(np.percentile(moving, 98)) if moving.size else 1.0
        self.vmax = float(SPEED_VMAX) if SPEED_VMAX else max(auto_vmax, 1.0)

        # Trials: maximal runs sharing one target position, same definition
        # used by fitts_analysis.segment_trials.
        changed = (np.diff(self.tx) != 0) | (np.diff(self.ty) != 0)
        self.block = np.concatenate([[0], np.cumsum(changed)]).astype(int)
        self.n_blocks = int(self.block[-1]) + 1
        self.block_start = np.zeros(self.n_blocks, dtype=int)
        self.block_end = np.zeros(self.n_blocks, dtype=int)
        for b in range(self.n_blocks):
            idx = np.flatnonzero(self.block == b)
            self.block_start[b] = idx[0]
            self.block_end[b] = idx[-1]
        self.block_success = np.array(
            [bool((self.hold[self.block_start[b]:self.block_end[b] + 1] >= HOLD_FRAMES).any())
             for b in range(self.n_blocks)])

        self.center = (SCREEN_SIZE[0] / 2.0, SCREEN_SIZE[1] / 2.0)

        # Conditions: one (amplitude, width) combination, that is one index of
        # difficulty. The runner exhausts max_targets trials on a combination
        # before advancing, so a condition is a contiguous run of trials and is
        # recovered by change detection on the per-trial (W, D) key. D is
        # quantized to COND_D_TOL px to absorb the integer rounding of the ring
        # points; the configured ring radii are far enough apart that no two
        # collapse. In mode A every target is at its own distance, so a
        # condition degenerates to a single trial.
        blk_w = self.rad[self.block_start]
        blk_d = np.hypot(self.tx[self.block_start] - self.center[0],
                         self.ty[self.block_start] - self.center[1])
        key = np.stack([blk_w, np.round(blk_d / COND_D_TOL)], axis=1)
        changed_c = (np.diff(key, axis=0) != 0).any(axis=1)
        self.block_cond = np.concatenate([[0], np.cumsum(changed_c)]).astype(int)
        self.n_conds = int(self.block_cond[-1]) + 1
        self.cond = self.block_cond[self.block]
        self.cond_start = np.zeros(self.n_conds, dtype=int)
        self.cond_end = np.zeros(self.n_conds, dtype=int)
        for c in range(self.n_conds):
            idx = np.flatnonzero(self.cond == c)
            self.cond_start[c] = idx[0]
            self.cond_end[c] = idx[-1]

        # Ring layout per target radius, reconstructed from the logged targets.
        self.rings = {}
        for r in np.unique(self.rad):
            m = self.rad == r
            pts = np.unique(np.stack([self.tx[m], self.ty[m]], axis=1), axis=0)
            self.rings[float(r)] = pts

    def condition(self, i):
        """Amplitude, width and index of difficulty of the current trial.
        W is the target diameter and D the distance from screen center, so ID
        follows the Shannon formulation used in fitts_analysis."""
        w = 2.0 * self.rad[i]
        d = math.hypot(self.tx[i] - self.center[0], self.ty[i] - self.center[1])
        idd = math.log2(d / w + 1.0) if w > 0 else float('nan')
        return d, w, idd

    def display_model(self):
        return DISPLAY_NAMES.get(self.model, self.model)


# ======== REPLAY VIEW ========

class ReplayView(QWidget):

    def __init__(self, session, subject):
        super().__init__()
        self.s = session
        self.subject = subject
        self.theme = THEME_PAPER if PAPER_MODE else THEME_DARK
        self.lut = build_lut(TRACE_CMAP)

        self.scale = float(VIEW_SCALE)
        self.setWindowTitle("Fitts Replay  |  subject %s  |  %s"
                            % (subject, self.s.display_model()))
        self.setFixedSize(int(SCREEN_SIZE[0] * self.scale),
                          int(SCREEN_SIZE[1] * self.scale))
        self.setFocusPolicy(Qt.StrongFocus)

        self.trace_on = TRACE_ENABLED
        self.trace_scope = TRACE_SCOPE
        self.color_by = TRACE_COLOR_BY

        self.idx = 0
        self.paused = bool(START_PAUSED)
        self.speed = float(INITIAL_SPEED)
        self.play_t = 0.0
        self.dashboard = None

        self.trace_pm = QPixmap(self.size())
        self.trace_pm.fill(Qt.transparent)
        self.drawn_to = 0
        self.scope_from = 0
        self.span = (0, self.s.n - 1)

        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(1, int(1000 / FRAME_RATE)))

        self.rebuild_trace()
        self.show()

    # -------- transport --------

    def tick(self):
        dt = self.clock.restart() / 1000.0
        if self.paused:
            return
        self.play_t += dt * self.speed
        i = self.idx
        while i + 1 < self.s.n and self.s.t[i + 1] <= self.play_t:
            i += 1
        if i != self.idx:
            self.set_index(i, from_playback=True)
        if self.idx >= self.s.n - 1:
            if LOOP:
                self.seek(0)
            else:
                self.paused = True
                if self.dashboard is not None:
                    self.dashboard.sync()

    def set_index(self, i, from_playback=False):
        i = int(max(0, min(i, self.s.n - 1)))
        backwards = i < self.idx
        self.idx = i
        if not from_playback:
            self.play_t = float(self.s.t[i])
        if self.trace_on:
            if backwards:
                self.rebuild_trace()
            else:
                self.extend_trace()
        self.update()
        if self.dashboard is not None:
            self.dashboard.sync()

    def seek(self, i):
        self.set_index(i)

    def step(self, delta):
        self.paused = True
        self.set_index(self.idx + delta)

    def jump_trial(self, delta):
        b = int(self.s.block[self.idx])
        # A step back inside a trial returns to its start before leaving it.
        if delta < 0 and self.idx > self.s.block_start[b] + 2:
            target = b
        else:
            target = b + delta
        target = int(max(0, min(target, self.s.n_blocks - 1)))
        self.paused = True
        self.set_index(int(self.s.block_start[target]))

    def jump_condition(self, delta):
        c = int(self.s.cond[self.idx])
        # A step back from inside a condition returns to its start first.
        if delta < 0 and self.idx > self.s.cond_start[c] + 2:
            target = c
        else:
            target = c + delta
        target = int(max(0, min(target, self.s.n_conds - 1)))
        self.paused = True
        self.set_index(int(self.s.cond_start[target]))

    def toggle_pause(self):
        self.paused = not self.paused
        self.clock.restart()
        if self.dashboard is not None:
            self.dashboard.sync()

    def set_speed(self, v):
        self.speed = float(v)

    # -------- trace cache --------

    def scope_span(self, i):
        """First and last frame index the current trace scope can ever cover.
        Fixed for the whole scope so the time colour ramp does not shift as
        segments are appended."""
        if self.trace_scope == 'session':
            return 0, self.s.n - 1
        if self.trace_scope == 'condition':
            c = int(self.s.cond[i])
            return int(self.s.cond_start[c]), int(self.s.cond_end[c])
        if self.trace_scope == 'trial':
            b = int(self.s.block[i])
            return int(self.s.block_start[b]), int(self.s.block_end[b])
        a = int(max(0, i - TRACE_WINDOW_FRAMES))
        return a, i

    def scope_start(self, i):
        if not self.trace_on:
            return i
        return self.scope_span(i)[0]

    def rebuild_trace(self):
        """Full redraw of the cached trace layer. Runs on seek, scope change
        and colour change only."""
        self.trace_pm.fill(Qt.transparent)
        if not self.trace_on:
            self.drawn_to = self.idx
            self.scope_from = self.idx
            return
        start = self.scope_start(self.idx)
        self.span = self.scope_span(self.idx)
        p = QPainter(self.trace_pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(self.scale, self.scale)
        if start > 0 and TRACE_KEEP_PAST and self.trace_scope in ('trial', 'condition'):
            self.draw_segments(p, 0, start, ghost=True)
        self.draw_segments(p, start, self.idx, ghost=False)
        p.end()
        self.scope_from = start
        self.drawn_to = self.idx

    def extend_trace(self):
        """Append only the new segments when the scope start has not moved."""
        if not self.trace_on:
            return
        start = self.scope_start(self.idx)
        if start != self.scope_from:
            self.rebuild_trace()
            return
        if self.idx <= self.drawn_to:
            return
        p = QPainter(self.trace_pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(self.scale, self.scale)
        self.draw_segments(p, self.drawn_to, self.idx, ghost=False)
        p.end()
        self.drawn_to = self.idx

    def seg_color(self, i):
        if self.color_by == 'speed':
            t = self.s.seg_speed[i] / self.s.vmax
            k = int(max(0.0, min(1.0, t)) * (len(self.lut) - 1))
            return self.lut[k]
        if self.color_by == 'gesture':
            return GESTURE_COLORS.get(int(self.s.gesture[i]), self.theme['flat_trace'])
        if self.color_by == 'time':
            a, b = self.span
            b = max(b, a + 1)
            t = (i - a) / float(b - a)
            k = int(max(0.0, min(1.0, t)) * (len(self.lut) - 1))
            return self.lut[k]
        return self.theme['flat_trace']

    def draw_segments(self, p, a, b, ghost=False):
        """Draw path segments [a, b) in logical task coordinates."""
        a = int(max(0, a))
        b = int(min(b, self.s.n - 1))
        if b <= a:
            return
        s = self.s
        if ghost:
            pen = QPen(self.theme['ghost'], TRACE_WIDTH)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            for i in range(a, b):
                p.drawLine(QPointF(s.cx[i], s.cy[i]), QPointF(s.cx[i + 1], s.cy[i + 1]))
        else:
            pen = QPen()
            pen.setWidthF(TRACE_WIDTH)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            last = None
            for i in range(a, b):
                col = self.seg_color(i)
                if col != last:
                    pen.setColor(col)
                    p.setPen(pen)
                    last = col
                p.drawLine(QPointF(s.cx[i], s.cy[i]), QPointF(s.cx[i + 1], s.cy[i + 1]))
        if TRACE_MARK_TRIALS:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self.theme['text']))
            for bi in range(s.n_blocks):
                k = int(s.block_start[bi])
                if a <= k <= b:
                    p.drawEllipse(QPointF(s.cx[k], s.cy[k]), 2.6, 2.6)
            p.setBrush(Qt.NoBrush)

    # -------- scene drawing --------

    def draw_targets(self, p, i):
        s = self.s
        r = float(s.rad[i])
        pen = QPen(self.theme['ring'], 1.6)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if s.mode == "B":
            for x, y in s.rings.get(r, []):
                p.drawEllipse(QPointF(float(x), float(y)), r, r)
        active = self.theme['target_hold'] if s.hold[i] > 0 else self.theme['target_idle']
        fill = QColor(active)
        fill.setAlpha(90 if PAPER_MODE else 255)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(active, 2.2))
        p.drawEllipse(QPointF(float(s.tx[i]), float(s.ty[i])), r, r)
        p.setBrush(Qt.NoBrush)

    def draw_cursor(self, p, i):
        s = self.s
        p.setPen(QPen(self.theme['cursor_edge'], 1.4))
        p.setBrush(QBrush(self.theme['cursor']))
        p.drawEllipse(QPointF(float(s.cx[i]), float(s.cy[i])), 6.0, 6.0)
        p.setBrush(Qt.NoBrush)

    # -------- overlays --------

    def overlay_scale(self, w):
        """Multiplier that keeps overlay geometry proportional to the render
        size: VIEW_SCALE on screen, SHOT_SCALE in an export."""
        return w / float(SCREEN_SIZE[0])

    def draw_colorbar(self, p, w, h):
        if not SHOW_COLORBAR or self.color_by == 'none':
            return
        f = self.overlay_scale(w)
        margin = COLORBAR_MARGIN * f
        title_pt = max(6, int(round(COLORBAR_TITLE_PT * f)))
        tick_pt = max(6, int(round(COLORBAR_FONT_PT * f)))

        if self.color_by == 'gesture':
            self.draw_gesture_legend(p, w, h, f, tick_pt)
            return

        if self.color_by == 'speed':
            title = "Cursor speed (px/s)"
            n_ticks = max(2, COLORBAR_TICKS)
            ticks = []
            for k in range(n_ticks):
                frac = k / float(n_ticks - 1)
                ticks.append((frac, "%d" % int(round(frac * self.s.vmax))))
        else:
            title = "Time in trace"
            ticks = [(0.0, "start"), (1.0, "now")]

        p.setFont(QFont("Arial", tick_pt))
        fm = p.fontMetrics()
        label_w = max(fm.horizontalAdvance(txt) for _, txt in ticks) + 10 * f
        tick_len = 9 * f
        gap = 12 * f

        if COLORBAR_ORIENT == 'vertical':
            bar_w = COLORBAR_THICKNESS * f
            bar_h = h * COLORBAR_LENGTH_FRAC
            x0 = w - margin - label_w - tick_len - gap - bar_w
            y0 = (h - bar_h) / 2.0
            grad = QLinearGradient(0.0, y0 + bar_h, 0.0, y0)
        else:
            bar_w = w * COLORBAR_LENGTH_FRAC * 0.5
            bar_h = COLORBAR_THICKNESS * f
            x0 = w - margin - bar_w
            y0 = h - margin - bar_h - tick_pt * 2.4
            grad = QLinearGradient(x0, 0.0, x0 + bar_w, 0.0)

        for k in range(33):
            grad.setColorAt(k / 32.0, self.lut[int(k / 32.0 * (len(self.lut) - 1))])

        if OVERLAY_PANEL:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self.theme['panel']))
            if COLORBAR_ORIENT == 'vertical':
                p.drawRoundedRect(QRectF(x0 - title_pt * 2.6 - gap, y0 - 18 * f,
                                         bar_w + tick_len + gap + label_w + title_pt * 2.6 + gap,
                                         bar_h + 36 * f), 6 * f, 6 * f)
            else:
                p.drawRoundedRect(QRectF(x0 - gap, y0 - title_pt * 2.4,
                                         bar_w + 2 * gap,
                                         bar_h + title_pt * 2.4 + tick_pt * 2.4), 6 * f, 6 * f)
            p.setBrush(Qt.NoBrush)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRect(QRectF(x0, y0, bar_w, bar_h))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self.theme['text'], max(1.0, 1.2 * f)))
        p.drawRect(QRectF(x0, y0, bar_w, bar_h))

        p.setFont(QFont("Arial", tick_pt))
        if COLORBAR_ORIENT == 'vertical':
            for frac, txt in ticks:
                y = y0 + bar_h * (1.0 - frac)
                p.drawLine(QPointF(x0 + bar_w, y), QPointF(x0 + bar_w + tick_len, y))
                p.drawText(QRectF(x0 + bar_w + tick_len + 6 * f, y - tick_pt * 1.2,
                                  label_w, tick_pt * 2.4),
                           Qt.AlignLeft | Qt.AlignVCenter, txt)
            # Axis title rotated to read bottom-to-top alongside the bar.
            p.save()
            p.setFont(QFont("Arial", title_pt, QFont.Bold))
            th = p.fontMetrics().height()
            p.translate(x0 - gap - th, y0 + bar_h)
            p.rotate(-90)
            p.drawText(QRectF(0, 0, bar_h, th), Qt.AlignCenter, title)
            p.restore()
        else:
            for frac, txt in ticks:
                x = x0 + bar_w * frac
                p.drawLine(QPointF(x, y0 + bar_h), QPointF(x, y0 + bar_h + tick_len))
                p.drawText(QRectF(x - label_w / 2.0, y0 + bar_h + tick_len,
                                  label_w, tick_pt * 2.2),
                           Qt.AlignHCenter | Qt.AlignVCenter, txt)
            p.setFont(QFont("Arial", title_pt, QFont.Bold))
            th = p.fontMetrics().height()
            p.drawText(QRectF(x0, y0 - th - 4 * f, bar_w, th),
                       Qt.AlignHCenter | Qt.AlignVCenter, title)

    def draw_gesture_legend(self, p, w, h, f, font_pt):
        items = [(GESTURE_NAMES[k], GESTURE_COLORS[k]) for k in sorted(GESTURE_COLORS)]
        sw = GESTURE_SWATCH * f
        gap = sw * 0.6
        p.setFont(QFont("Arial", font_pt))
        label_w = max(p.fontMetrics().horizontalAdvance(name) for name, _ in items) + 12 * f
        box_h = len(items) * (sw + gap) - gap
        x0 = w - COLORBAR_MARGIN * f - label_w - sw
        y0 = (h - box_h) / 2.0
        if OVERLAY_PANEL:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self.theme['panel']))
            p.drawRoundedRect(QRectF(x0 - 12 * f, y0 - 12 * f,
                                     sw + label_w + 24 * f, box_h + 24 * f), 6 * f, 6 * f)
            p.setBrush(Qt.NoBrush)
        for j, (name, col) in enumerate(items):
            y = y0 + j * (sw + gap)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(x0, y, sw, sw))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self.theme['text']))
            p.drawText(QRectF(x0 + sw + 10 * f, y, label_w, sw),
                       Qt.AlignLeft | Qt.AlignVCenter, name)

    def draw_annotation(self, p, i, w, h):
        if not SHOW_ANNOTATION:
            return
        f = self.overlay_scale(w)
        s = self.s
        d, wd, idd = s.condition(i)
        b = int(s.block[i])
        c = int(s.cond[i])

        title_pt = max(7, int(round(ANNOTATION_TITLE_PT * f)))
        body_pt = max(6, int(round(ANNOTATION_BODY_PT * f)))
        pad = 14 * f
        x0 = ANNOTATION_MARGIN * f
        y0 = ANNOTATION_MARGIN * f

        ftitle = QFont("Arial", title_pt, QFont.Bold)
        fbody = QFont("Arial", body_pt)

        rows = [
            (ftitle, "Subject %s" % self.subject),
            (ftitle, "Model %s" % s.display_model()),
            (fbody, "Condition %d/%d" % (c + 1, s.n_conds)),
            (fbody, "D = %d px   W = %d px" % (int(round(d)), int(round(wd)))),
            (fbody, "ID = %.2f bits" % idd),
            (fbody, "Trial %d/%d   t = %.1f s" % (b + 1, s.n_blocks, s.t[i])),
        ]

        box_w = 0.0
        box_h = 0.0
        heights = []
        for font, txt in rows:
            p.setFont(font)
            fm = p.fontMetrics()
            box_w = max(box_w, fm.horizontalAdvance(txt))
            heights.append(fm.height())
            box_h += fm.height()
        box_w += 2 * pad
        box_h += 2 * pad + 4 * f * (len(rows) - 1)

        if OVERLAY_PANEL:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self.theme['panel']))
            p.drawRoundedRect(QRectF(x0, y0, box_w, box_h), 6 * f, 6 * f)
            p.setBrush(Qt.NoBrush)

        p.setPen(QPen(self.theme['text']))
        y = y0 + pad
        for (font, txt), rh in zip(rows, heights):
            p.setFont(font)
            p.drawText(QRectF(x0 + pad, y, box_w, rh),
                       Qt.AlignLeft | Qt.AlignVCenter, txt)
            y += rh + 4 * f

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.theme['bg'])
        p.save()
        p.scale(self.scale, self.scale)
        self.draw_targets(p, self.idx)
        p.restore()
        if self.trace_on:
            p.drawPixmap(0, 0, self.trace_pm)
        p.save()
        p.scale(self.scale, self.scale)
        self.draw_cursor(p, self.idx)
        p.restore()
        self.draw_colorbar(p, self.width(), self.height())
        self.draw_annotation(p, self.idx, self.width(), self.height())
        p.end()

    # -------- export --------

    def render_figure(self, scale=SHOT_SCALE, whole_session=False, with_cursor=True):
        """Render a standalone high-resolution frame, independent of the
        on-screen scaling and of the cached trace layer."""
        w = int(SCREEN_SIZE[0] * scale)
        h = int(SCREEN_SIZE[1] * scale)
        pm = QPixmap(w, h)
        pm.fill(self.theme['bg'])
        live_span = self.span
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.save()
        p.scale(scale, scale)
        self.draw_targets(p, self.idx)
        if self.trace_on:
            if whole_session:
                self.span = (0, self.s.n - 1)
                self.draw_segments(p, 0, self.s.n - 1, ghost=False)
            else:
                self.span = self.scope_span(self.idx)
                start = self.scope_start(self.idx)
                if start > 0 and TRACE_KEEP_PAST and self.trace_scope in ('trial', 'condition'):
                    self.draw_segments(p, 0, start, ghost=True)
                self.draw_segments(p, start, self.idx, ghost=False)
        if with_cursor:
            self.draw_cursor(p, self.idx)
        p.restore()
        self.draw_colorbar(p, w, h)
        self.draw_annotation(p, self.idx, w, h)
        p.end()
        self.span = live_span
        return pm

    def save_shot(self, whole_session=False):
        os.makedirs(SHOT_DIR, exist_ok=True)
        tag = "session" if whole_session else ("%s_f%06d" % (self.trace_scope, self.idx))
        name = "%s_%s_%s_%s.png" % (self.subject, self.s.model, self.color_by, tag)
        path = join(SHOT_DIR, name)
        pm = self.render_figure(whole_session=whole_session,
                                with_cursor=not whole_session)
        pm.save(path)
        return path

    # -------- input --------

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space:
            self.toggle_pause()
        elif k == Qt.Key_Right:
            self.step(1)
        elif k == Qt.Key_Left:
            self.step(-1)
        elif k == Qt.Key_Period:
            self.jump_trial(1)
        elif k == Qt.Key_Comma:
            self.jump_trial(-1)
        elif k == Qt.Key_BracketRight:
            self.jump_condition(1)
        elif k == Qt.Key_BracketLeft:
            self.jump_condition(-1)
        elif k == Qt.Key_T:
            self.trace_on = not self.trace_on
            self.rebuild_trace()
            self.update()
            if self.dashboard is not None:
                self.dashboard.sync()
        elif k == Qt.Key_S:
            path = self.save_shot()
            print("Saved %s" % path)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.dashboard is not None:
            self.dashboard.close()
        event.accept()


# ======== DASHBOARD ========

class ReplayDashboard(QWidget):

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.view.dashboard = self
        self._syncing = False
        self.setWindowTitle("Replay Control")
        self.setFixedWidth(300)

        layout = QVBoxLayout()

        # -------- transport --------
        transport = QGroupBox("Transport")
        tl = QVBoxLayout()

        row = QHBoxLayout()
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self.view.toggle_pause)
        row.addWidget(self.play_btn)
        self.restart_btn = QPushButton("Restart")
        self.restart_btn.clicked.connect(lambda: self.view.seek(0))
        row.addWidget(self.restart_btn)
        tl.addLayout(row)

        row = QHBoxLayout()
        for text, fn in (("<< Trial", lambda: self.view.jump_trial(-1)),
                         ("- Frame", lambda: self.view.step(-1)),
                         ("+ Frame", lambda: self.view.step(1)),
                         ("Trial >>", lambda: self.view.jump_trial(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        tl.addLayout(row)

        row = QHBoxLayout()
        for text, fn in (("<< ID", lambda: self.view.jump_condition(-1)),
                         ("ID >>", lambda: self.view.jump_condition(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        tl.addLayout(row)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 16.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(self.view.speed)
        self.speed_spin.setPrefix("Speed: x")
        self.speed_spin.valueChanged.connect(self.view.set_speed)
        tl.addWidget(self.speed_spin)

        row = QHBoxLayout()
        for v in (0.25, 0.5, 1.0, 2.0, 4.0):
            b = QPushButton("x%g" % v)
            b.clicked.connect(lambda _=False, s=v: self.speed_spin.setValue(s))
            row.addWidget(b)
        tl.addLayout(row)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, self.view.s.n - 1)
        self.scrub.valueChanged.connect(self.on_scrub)
        tl.addWidget(QLabel("Position"))
        tl.addWidget(self.scrub)

        transport.setLayout(tl)
        layout.addWidget(transport)

        # -------- trace --------
        trace = QGroupBox("Trace")
        gl = QVBoxLayout()

        self.trace_check = QCheckBox("Show trace")
        self.trace_check.setChecked(self.view.trace_on)
        self.trace_check.toggled.connect(self.on_trace_toggle)
        gl.addWidget(self.trace_check)

        self.scope_box = QComboBox()
        self.scope_box.addItems(["session", "condition", "trial", "window"])
        self.scope_box.setCurrentText(self.view.trace_scope)
        self.scope_box.currentTextChanged.connect(self.on_scope)
        gl.addWidget(QLabel("Scope"))
        gl.addWidget(self.scope_box)

        self.color_box = QComboBox()
        self.color_box.addItems(["speed", "gesture", "time", "none"])
        self.color_box.setCurrentText(self.view.color_by)
        self.color_box.currentTextChanged.connect(self.on_color)
        gl.addWidget(QLabel("Colour by"))
        gl.addWidget(self.color_box)

        row = QHBoxLayout()
        shot_btn = QPushButton("Save frame")
        shot_btn.clicked.connect(lambda: self.on_save(False))
        row.addWidget(shot_btn)
        full_btn = QPushButton("Save full path")
        full_btn.clicked.connect(lambda: self.on_save(True))
        row.addWidget(full_btn)
        gl.addLayout(row)

        trace.setLayout(gl)
        layout.addWidget(trace)

        # -------- readout --------
        info = QGroupBox("Session")
        il = QGridLayout()
        self.labels = {}
        keys = ["file", "model", "frame", "time", "trial", "condition",
                "id", "gesture", "speed", "hold", "outcome"]
        for r, k in enumerate(keys):
            il.addWidget(QLabel("<b>%s</b>" % k.capitalize()), r, 0)
            lab = QLabel("-")
            lab.setWordWrap(True)
            self.labels[k] = lab
            il.addWidget(lab, r, 1)
        info.setLayout(il)
        layout.addWidget(info)

        layout.addWidget(QLabel("Space pause   Arrows step   , . trial   "
                                "[ ] condition   T trace   S screenshot"))
        layout.addStretch(1)
        self.setLayout(layout)

        # Same bindings as the replay window, so focus can sit on either one.
        self.shortcuts = []
        for seq, fn in (("Space", self.view.toggle_pause),
                        ("Right", lambda: self.view.step(1)),
                        ("Left", lambda: self.view.step(-1)),
                        (",", lambda: self.view.jump_trial(-1)),
                        (".", lambda: self.view.jump_trial(1)),
                        ("[", lambda: self.view.jump_condition(-1)),
                        ("]", lambda: self.view.jump_condition(1)),
                        ("S", lambda: self.on_save(False))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)
            self.shortcuts.append(sc)

        self.labels["file"].setText(os.path.basename(self.view.s.path))
        self.labels["model"].setText(self.view.s.model)
        self.show()
        self.sync()

    # -------- handlers --------

    def on_scrub(self, v):
        if self._syncing:
            return
        self.view.paused = True
        self.view.seek(int(v))

    def on_trace_toggle(self, on):
        self.view.trace_on = bool(on)
        self.view.rebuild_trace()
        self.view.update()

    def on_scope(self, text):
        self.view.trace_scope = text
        self.view.rebuild_trace()
        self.view.update()

    def on_color(self, text):
        self.view.color_by = text
        self.view.rebuild_trace()
        self.view.update()

    def on_save(self, whole_session):
        path = self.view.save_shot(whole_session=whole_session)
        print("Saved %s" % path)
        self.labels["file"].setText(os.path.basename(path))

    # -------- readout --------

    def sync(self):
        v = self.view
        s = v.s
        i = v.idx
        self._syncing = True
        self.scrub.setValue(i)
        self._syncing = False
        self.play_btn.setText("Resume" if v.paused else "Pause")
        self.trace_check.setChecked(v.trace_on)

        b = int(s.block[i])
        c = int(s.cond[i])
        d, w, idd = s.condition(i)
        seg = min(i, s.n - 2)
        self.labels["frame"].setText("%d / %d" % (i + 1, s.n))
        self.labels["time"].setText("%.2f s / %.2f s" % (s.t[i], s.t[-1]))
        self.labels["trial"].setText("%d / %d" % (b + 1, s.n_blocks))
        self.labels["condition"].setText("%d / %d   D %d px, W %d px"
                                         % (c + 1, s.n_conds, int(round(d)), int(round(w))))
        self.labels["id"].setText("%.2f bits" % idd)
        self.labels["gesture"].setText(GESTURE_NAMES.get(int(s.gesture[i]), "?"))
        self.labels["speed"].setText("%.0f px/s" % s.seg_speed[seg])
        self.labels["hold"].setText("%d / %d" % (int(s.hold[i]), HOLD_FRAMES))
        self.labels["outcome"].setText("acquired" if s.block_success[b] else "timeout")

    def closeEvent(self, event):
        event.accept()


# ======== MAIN ========

def main():
    path = LOG_PATH if LOG_PATH else find_log(FITTS_ROOT, USER_ID, MODEL, SESSION_INDEX)
    if not exists(path):
        raise FileNotFoundError("Log not found: %s" % path)
    df = load_log(path)
    session = Session(df, path)
    subject = str(USER_ID) if not LOG_PATH else os.path.basename(os.path.dirname(path))

    print("Replaying %s" % path)
    print("Frames: %d   Trials: %d   Conditions: %d   Duration: %.1f s   "
          "Speed vmax: %.0f px/s"
          % (session.n, session.n_blocks, session.n_conds, session.t[-1],
             session.vmax))
    for c in range(session.n_conds):
        k = int(session.cond_start[c])
        d, w, idd = session.condition(k)
        n_tr = int(session.block[session.cond_end[c]] - session.block[k] + 1)
        print("  ID %d: D = %d px   W = %d px   %.2f bits   %d trials"
              % (c + 1, int(round(d)), int(round(w)), idd, n_tr))

    app = QApplication(sys.argv)
    view = ReplayView(session, subject)
    dash = ReplayDashboard(view)
    dash.move(view.x() + view.width() + 12, view.y())
    view.setFocus()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()