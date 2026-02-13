import numpy as np
import time
from copy import deepcopy
import os
import sys
import csv
import math
import threading
import random
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from utils import *


VEL_CONSTANT = 5.0


# Shared Context for Thread Safety and Global State
class SharedContext:
    def __init__(self):
        self.lock = threading.Lock()
        
        # EMG State
        self.emg_x = 0.0
        self.emg_y = 0.0
        self.probs = []
        self.raw_velocity = 0.0
        
        # Model & Control Settings
        self.active_model_name = None
        self.available_models = []
        self.flip_lr = False
        self.speed_multiplier = 1.0
        
        # Fitts Parameters
        self.params = {
            'frame_rate': 60,
            'mode': 'B',
            'hold_frames_required': 180,
            'target_timeout_frames': 900,
            'max_targets': 16,
            'target_radius_range': [30, 30],
            'target_distance_range': [200, 400],
            'ring_radius': 250,
            'velocity_scale': 5.0,
            'physics': {
                'enabled': False,
                'mass': 5,
                'max_acceleration': 0.08,
                'damping': 1,
            },
            'screen_size': (1200, 800),
            'c_vel': 1,
            'use_test_input': False,
        }

    def update_emg(self, x, y, probs, velocity):
        with self.lock:
            self.emg_x = x
            self.emg_y = y
            self.probs = probs
            self.raw_velocity = velocity

    def get_emg(self):
        with self.lock:
            return self.emg_x, self.emg_y, self.probs, self.raw_velocity

    def set_param(self, key, value):
        with self.lock:
            self.params[key] = value

    def set_physics_param(self, key, value):
        with self.lock:
            self.params['physics'][key] = value

    def get_params(self):
        with self.lock:
            return deepcopy(self.params)  # full copy including nested dicts

    def set_models(self, available_models):
        with self.lock:
            self.available_models = available_models
            if not available_models:
                raise RuntimeError("No models provided")
            self.active_model_name = available_models[0]
    
    def set_model(self, active_model_name):
        with self.lock:
            self.active_model_name = active_model_name

    def get_model(self):
        with self.lock:
            return self.active_model_name

# Global instance to be imported by main.py
shared = SharedContext()

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fitts' Law Dashboard")
        self.test_window = None
        self.test_input_panel = None
        
        layout = QVBoxLayout()
        
        # --- Model Selection ---
        layout.addWidget(QLabel("<b>Model Selection</b>"))
        self.model_box = QComboBox()
        self.model_box.currentTextChanged.connect(self.update_model)
        layout.addWidget(self.model_box)

        # --- Control Settings ---
        control_layout = QHBoxLayout()
        self.flip_check = QCheckBox("Flip L/R (Left-Handed)")
        self.flip_check.toggled.connect(self.update_controls)
        control_layout.addWidget(self.flip_check)
        
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setPrefix("Speed: x")
        self.speed_spin.valueChanged.connect(self.update_controls)
        control_layout.addWidget(self.speed_spin)
        layout.addLayout(control_layout)
        
        layout.addWidget(QLabel("<hr>"))

        # --- Fitts Mode ---
        self.mode_box = QComboBox()
        self.mode_box.addItems(["A", "B", "C"])
        self.mode_box.setCurrentIndex(1)
        self.mode_box.currentTextChanged.connect(self.update_params)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_box)

        # --- Parameters ---
        params = shared.params # Initial read
        
        self.fps = QSpinBox()
        self.fps.setRange(10, 165)
        self.fps.setValue(params['frame_rate'])
        self.fps.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Frame Rate:"))
        layout.addWidget(self.fps)

        self.hold = QSpinBox()
        self.hold.setRange(1, 3600)
        self.hold.setValue(params['hold_frames_required'])
        self.hold.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Hold Frames (A/B):"))
        layout.addWidget(self.hold)

        self.max_targets = QSpinBox()
        self.max_targets.setRange(1, 100)
        self.max_targets.setValue(params['max_targets'])
        self.max_targets.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Targets to Achieve (A/B):"))
        layout.addWidget(self.max_targets)

        self.timeout_frames = QSpinBox()
        self.timeout_frames.setRange(10, 216000)
        self.timeout_frames.setValue(params['target_timeout_frames'])
        self.timeout_frames.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Target Timeout (Frames):"))
        layout.addWidget(self.timeout_frames)

        # --- Physics ---
        self.physics_enabled = QCheckBox("Enable Physics")
        self.physics_enabled.setChecked(params['physics']['enabled'])
        self.physics_enabled.toggled.connect(self.update_params)
        layout.addWidget(self.physics_enabled)
        
        p_layout = QHBoxLayout()
        self.mass = QDoubleSpinBox(); self.mass.setValue(params['physics']['mass'])
        self.mass.setPrefix("Mass: "); self.mass.valueChanged.connect(self.update_params)
        p_layout.addWidget(self.mass)
        
        self.damping = QDoubleSpinBox(); self.damping.setValue(params['physics']['damping'])
        self.damping.setRange(0, 1); self.damping.setSingleStep(0.01)
        self.damping.setPrefix("Damp: "); self.damping.valueChanged.connect(self.update_params)
        p_layout.addWidget(self.damping)

        layout.addLayout(p_layout)

        # --- Geometry ---
        self.radius_min = QSpinBox(); self.radius_max = QSpinBox()
        self.radius_min.setRange(5, 300); self.radius_max.setRange(5, 300)
        self.radius_min.setValue(params['target_radius_range'][0])
        self.radius_max.setValue(params['target_radius_range'][1])
        self.radius_min.valueChanged.connect(self.update_params)
        self.radius_max.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Target Radius Range (px):"))
        r_layout = QHBoxLayout(); r_layout.addWidget(self.radius_min); r_layout.addWidget(self.radius_max)
        layout.addLayout(r_layout)

        self.dist_min = QSpinBox(); self.dist_max = QSpinBox()
        self.dist_min.setRange(10, 1000); self.dist_max.setRange(10, 1000)
        self.dist_min.setValue(params['target_distance_range'][0])
        self.dist_max.setValue(params['target_distance_range'][1])
        self.dist_min.valueChanged.connect(self.update_params)
        self.dist_max.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Target Distance Range (Mode A):"))
        d_layout = QHBoxLayout(); d_layout.addWidget(self.dist_min); d_layout.addWidget(self.dist_max)
        layout.addLayout(d_layout)

        self.ring_radius = QSpinBox()
        self.ring_radius.setRange(50, 1000)
        self.ring_radius.setValue(params['ring_radius'])
        self.ring_radius.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Ring Radius (Mode B):"))
        layout.addWidget(self.ring_radius)
        
        self.c_vel = QDoubleSpinBox()
        self.c_vel.setValue(params['c_vel'])
        self.c_vel.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Mode C Velocity:"))
        layout.addWidget(self.c_vel)

        self.test_input_checkbox = QCheckBox("Use Test Input Panel (X;Y)")
        self.test_input_checkbox.setChecked(params['use_test_input'])
        self.test_input_checkbox.toggled.connect(self.update_params)
        layout.addWidget(self.test_input_checkbox)

        self.launch_btn = QPushButton("Launch / Update Test")
        self.launch_btn.clicked.connect(self.launch_test)
        layout.addWidget(self.launch_btn)

        self.setLayout(layout)
        self.show()

    def update_model_list(self, model_names):
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(model_names)
        self.model_box.blockSignals(False)
        if shared.active_model_name in model_names:
            self.model_box.setCurrentText(shared.active_model_name)

    def update_model(self, text):
        shared.set_model(text)
        print(f"Model switched to: {text}")

    def update_controls(self):
        with shared.lock:
            shared.flip_lr = self.flip_check.isChecked()
            shared.speed_multiplier = self.speed_spin.value()

    def update_params(self):
        with shared.lock:
            p = shared.params
            p['frame_rate'] = self.fps.value()
            p['mode'] = self.mode_box.currentText()
            p['hold_frames_required'] = self.hold.value()
            p['max_targets'] = self.max_targets.value()
            p['target_timeout_frames'] = self.timeout_frames.value()
            
            p['physics']['mass'] = self.mass.value()
            p['physics']['damping'] = self.damping.value()
            p['physics']['enabled'] = self.physics_enabled.isChecked()
            
            p['target_radius_range'] = [self.radius_min.value(), self.radius_max.value()]
            if p['target_radius_range'][0] >= p['target_radius_range'][1]:
                 p['target_radius_range'][1] = p['target_radius_range'][0] + 1
                 
            p['target_distance_range'] = [self.dist_min.value(), self.dist_max.value()]
            if p['target_distance_range'][0] >= p['target_distance_range'][1]:
                 p['target_distance_range'][1] = p['target_distance_range'][0] + 1
                 
            p['ring_radius'] = self.ring_radius.value()
            p['c_vel'] = self.c_vel.value()
            p['use_test_input'] = self.test_input_checkbox.isChecked()

        if shared.params['use_test_input']:
            if self.test_input_panel is None:
                self.test_input_panel = TestInputPanel()
            self.test_input_panel.show()
        elif self.test_input_panel is not None:
            self.test_input_panel.close()

    def launch_test(self):
        self.update_params()
        if self.test_window is None or not self.test_window.isVisible():
            self.test_window = FittsTest()
            self.test_window.dashboard = self
        else:
            self.test_window.raise_()
            self.test_window.activateWindow()

class TestInputPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Input Panel")
        layout = QVBoxLayout()
        self.x_input = QLineEdit("0.0")
        self.y_input = QLineEdit("0.0")
        layout.addWidget(QLabel("X Velocity (-1 to 1):"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y Velocity (-1 to 1):"))
        layout.addWidget(self.y_input)
        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.write_inputs)
        self.timer.start(50)
        self.show()

    def write_inputs(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            shared.update_emg(x, y, [0.2]*5, 0.0)
        except:
            pass
    
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

class FittsTest(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fitts' Law Test")
        params = shared.get_params()
        self.setFixedSize(*params['screen_size'])
        self.cursor_pos = [self.width() // 2, self.height() // 2]
        self.actual_velocity = [0.0, 0.0]
        self.hold_counter = 0
        self.targets_hit = 0
        self.target_timer = 0
        
        self.ring_sequence = [
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
            8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7
        ]
        self.ring_index = 24
        self.dashboard = None
        
        self.init_logger()
        self.init_target()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // params['frame_rate'])
        self.show()

    def init_logger(self):
        fitts_folder = FITTS_PATH
        if not os.path.exists(fitts_folder):
            os.mkdir(fitts_folder)
        
        ts = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
        model_tag = shared.active_model_name.replace(" ", "_")
        filename = f"{fitts_folder}/Fitts_{model_tag}_{ts}.csv"
        
        self.log_file = open(filename, "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(["time", "frame", "mode", "model", "cursor_x", "cursor_y", "target_x", "target_y",
                             "radius", "X", "Y", "vx", "vy", "acc_x", "acc_y", "inside", "hold_count", "velocity",
                             "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"])
        self.frame_count = 0

    def init_target(self):
        params = shared.get_params()
        mode = params['mode']
        w, h = self.width(), self.height()
        self.target_timer = 0

        if mode == "A":
            dist = random.randint(*params['target_distance_range'])
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * dist
            dy = math.sin(angle) * dist
            cx, cy = self.cursor_pos
            raw_x = int(cx + dx); raw_y = int(cy + dy)

            radius = random.randint(*params['target_radius_range'])
            x = max(radius, min(w - radius, raw_x))
            y = max(radius, min(h - radius, raw_y))
            self.target_radius = radius
            self.target_pos = [x, y]

        elif mode == "B":
            self.points = []
            cx, cy = self.width() // 2, self.height() // 2
            self.target_radius = random.randint(*params['target_radius_range'])
            for i in range(16):
                angle = 2 * math.pi * i / 16
                x = int(cx + params['ring_radius'] * math.cos(angle))
                y = int(cy + params['ring_radius'] * math.sin(angle))
                self.points.append((x, y))
            self.target_index = self.ring_sequence[self.ring_index % len(self.ring_sequence)]
            self.ring_index += 1
            self.target_pos = list(self.points[self.target_index])

        elif mode == "C":
            self.target_radius = random.randint(*params['target_radius_range'])
            x = random.randint(self.target_radius, w - self.target_radius)
            y = random.randint(self.target_radius, h - self.target_radius)
            self.target_pos = [x, y]
            self.target_velocity = [random.choice([-1, 1]) * params['c_vel'], 
                                    random.choice([-1, 1]) * params['c_vel']]

    def update_frame(self):
        params = shared.get_params()
        x_in, y_in, probs, velocity_raw = shared.get_emg()
        
        self.frame_count += 1
        self.target_timer += 1

        if params['physics']['enabled']:
            acc_x = (x_in - self.actual_velocity[0]) / params['physics']['mass']
            acc_y = (y_in - self.actual_velocity[1]) / params['physics']['mass']
            acc_x = max(-params['physics']['max_acceleration'], min(params['physics']['max_acceleration'], acc_x))
            acc_y = max(-params['physics']['max_acceleration'], min(params['physics']['max_acceleration'], acc_y))
            self.actual_velocity[0] += acc_x
            self.actual_velocity[1] += acc_y
            self.actual_velocity[0] *= params['physics']['damping']
            self.actual_velocity[1] *= params['physics']['damping']
        else:
            acc_x, acc_y = 0, 0
            self.actual_velocity[0] = x_in
            self.actual_velocity[1] = y_in

        self.cursor_pos[0] += self.actual_velocity[0]
        self.cursor_pos[1] += self.actual_velocity[1]
        self.cursor_pos[0] = max(0, min(self.cursor_pos[0], self.width()))
        self.cursor_pos[1] = max(0, min(self.cursor_pos[1], self.height()))

        dist = math.hypot(self.cursor_pos[0] - self.target_pos[0], self.cursor_pos[1] - self.target_pos[1])
        inside = dist <= self.target_radius
        
        if inside:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        if params['mode'] in ["A", "B"]:
            if self.hold_counter >= params['hold_frames_required'] or self.target_timer >= params['target_timeout_frames']:
                self.targets_hit += 1
                if self.targets_hit >= params['max_targets']:
                    self.close()
                    return 
                self.init_target()

        elif params['mode'] == "C":
            for i in range(2):
                self.target_pos[i] += self.target_velocity[i]
                max_val = self.width() if i == 0 else self.height()
                if self.target_pos[i] <= self.target_radius or self.target_pos[i] >= max_val - self.target_radius:
                    self.target_velocity[i] *= -1
                    self.target_pos[i] = max(self.target_radius, min(max_val - self.target_radius, self.target_pos[i]))

        self.logger.writerow([
            time.time(), self.frame_count, params['mode'], shared.active_model_name,
            *self.cursor_pos, *self.target_pos, self.target_radius, 
            x_in, y_in, *self.actual_velocity, acc_x, acc_y, int(inside), 
            self.hold_counter, velocity_raw, *probs
        ])
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#252525"))
        qp.setRenderHint(QPainter.Antialiasing)
        
        params = shared.get_params()

        if params['mode'] == "B":
            for i, (x, y) in enumerate(self.points):
                qp.setBrush(Qt.NoBrush)
                qp.setPen(QPen(Qt.black, 2))
                qp.drawEllipse(int(x - self.target_radius), int(y - self.target_radius),
                               2 * self.target_radius, 2 * self.target_radius)
            x, y = self.target_pos
            qp.setBrush(QBrush(QColor("#10DA39") if self.hold_counter > 0 else QColor("#BD1B1B")))
            qp.drawEllipse(int(x - self.target_radius), int(y - self.target_radius),
                           2 * self.target_radius, 2 * self.target_radius)
        else:
            qp.setBrush(QBrush(QColor("#10DA39") if self.hold_counter > 0 else QColor("#BD1B1B")))
            qp.drawEllipse(int(self.target_pos[0] - self.target_radius),
                           int(self.target_pos[1] - self.target_radius),
                           2 * self.target_radius, 2 * self.target_radius)

        qp.setBrush(QBrush(QColor("#10C2DA")))
        qp.drawEllipse(int(self.cursor_pos[0]) - 5, int(self.cursor_pos[1]) - 5, 10, 10)

    def closeEvent(self, event):
        self.log_file.close()
        self.timer.stop()
        if self.dashboard:
            self.dashboard.test_window = None 
        event.accept()