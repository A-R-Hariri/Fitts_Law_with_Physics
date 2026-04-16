import numpy as np
import time
from copy import deepcopy
import os
import csv
import math
import threading
import random
from datetime import datetime
from multiprocessing import Manager

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, QLineEdit, QMessageBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from utils import *

VEL_CONSTANT = 20.0


# Placeholder for the proxy object initialized in main.py
SharedContext = None

class Dashboard(QWidget):
    def __init__(self, shared_context):
        super().__init__()
        self.sc = shared_context
        self.setWindowTitle("Fitts' Law Dashboard")
        self.test_window = None
        self.test_input_panel = None
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Model Selection</b>"))
        self.model_box = QComboBox()
        self.model_box.currentTextChanged.connect(self.update_model)
        layout.addWidget(self.model_box)

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

        self.mode_box = QComboBox()
        self.mode_box.addItems(["A", "B", "C"])
        self.mode_box.setCurrentIndex(1)
        self.mode_box.currentTextChanged.connect(self.update_params)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_box)

        params = self.sc.params
        
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

        self.radius_list_input = QLineEdit(",".join(map(str, params.get('target_radius_list', [20]))))
        self.radius_list_input.textChanged.connect(self.update_params)
        layout.addWidget(QLabel("Target Radius List (px, comma separated):"))
        layout.addWidget(self.radius_list_input)

        self.ring_radius_list_input = QLineEdit(",".join(map(str, params.get('ring_radius_list', [300]))))
        self.ring_radius_list_input.textChanged.connect(self.update_params)
        layout.addWidget(QLabel("Ring Radius List (Mode B, comma separated):"))
        layout.addWidget(self.ring_radius_list_input)
        
        self.c_vel = QDoubleSpinBox()
        self.c_vel.setValue(params['c_vel'])
        self.c_vel.valueChanged.connect(self.update_params)
        layout.addWidget(QLabel("Mode C Velocity:"))
        layout.addWidget(self.c_vel)

        # self.test_input_checkbox = QCheckBox("Use Test Input Panel (X;Y)")
        # self.test_input_checkbox.setChecked(params['use_test_input'])
        # self.test_input_checkbox.toggled.connect(self.update_params)
        # layout.addWidget(self.test_input_checkbox)

        self.snap_checkbox = QCheckBox("Snap back")
        self.snap_checkbox.setChecked(params['snap_back'])
        self.snap_checkbox.toggled.connect(self.update_params)
        layout.addWidget(self.snap_checkbox)

        self.test_checkbox = QCheckBox("Test Run")
        self.test_checkbox.setChecked(True)
        layout.addWidget(self.test_checkbox)

        self.launch_btn = QPushButton("Launch / Update Test")
        self.launch_btn.clicked.connect(self.launch_test)
        layout.addWidget(self.launch_btn)

        self.stop_btn = QPushButton("Stop Test")
        self.stop_btn.clicked.connect(self.stop_test)
        layout.addWidget(self.stop_btn)
        self.stop_btn.setEnabled(False)

        self.setLayout(layout)
        self.show()

    def update_model_list(self, model_names):
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(model_names)
        self.model_box.blockSignals(False)
        if self.sc.active_model_name in model_names:
            self.model_box.setCurrentText(self.sc.active_model_name)

    def update_model(self, text):
        self.sc.active_model_name = text

    def update_controls(self):
        self.sc.flip_lr = self.flip_check.isChecked()
        self.sc.speed_multiplier = self.speed_spin.value()

    def update_params(self):
        p = deepcopy(self.sc.params)
        p['frame_rate'] = self.fps.value()
        p['mode'] = self.mode_box.currentText()
        p['hold_frames_required'] = self.hold.value()
        p['max_targets'] = self.max_targets.value()
        p['target_timeout_frames'] = self.timeout_frames.value()
        p['physics']['mass'] = self.mass.value()
        p['physics']['damping'] = self.damping.value()
        p['physics']['enabled'] = self.physics_enabled.isChecked()

        try:
            p['target_radius_list'] = [int(x.strip()) for x in self.radius_list_input.text().split(',') if x.strip().isdigit()]
            if not p['target_radius_list']: p['target_radius_list'] = [200]
        except: p['target_radius_list'] = [200]

        try:
            p['ring_radius_list'] = [int(x.strip()) for x in self.ring_radius_list_input.text().split(',') if x.strip().isdigit()]
            if not p['ring_radius_list']: p['ring_radius_list'] = [300]
        except: p['ring_radius_list'] = [300]

        p['c_vel'] = self.c_vel.value()
        # p['use_test_input'] = self.test_input_checkbox.isChecked()
        p['snap_back'] = self.snap_checkbox.isChecked()
        self.sc.params = p

        if p['use_test_input']:
            if self.test_input_panel is None:
                self.test_input_panel = TestInputPanel(self.sc)
            self.test_input_panel.show()
        elif self.test_input_panel is not None:
            self.test_input_panel.close()

    def launch_test(self):
        self.update_params()
        if self.test_window is None or not self.test_window.isVisible():
            self.test_window = FittsTest(self.sc, self)
            self.stop_btn.setEnabled(True)
        else:
            self.test_window.raise_()
            self.test_window.activateWindow()

    def stop_test(self):
        if self.test_window is not None or self.test_window.isVisible():
            self.test_window.close()
            self.stop_btn.setEnabled(False)
            self.test_window = None

# class TestInputPanel(QWidget):
#     def __init__(self, shared_context):
#         super().__init__()
#         self.sc = shared_context
#         self.setWindowTitle("Test Input Panel")
#         layout = QVBoxLayout()
#         self.x_input = QLineEdit("0.0")
#         self.y_input = QLineEdit("0.0")
#         layout.addWidget(QLabel("X Velocity (-1 to 1):"))
#         layout.addWidget(self.x_input)
#         layout.addWidget(QLabel("Y Velocity (-1 to 1):"))
#         layout.addWidget(self.y_input)
#         self.setLayout(layout)
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.write_inputs)
#         self.timer.start(50)
#         self.show()

#     def write_inputs(self):
#         try:
#             self.sc.emg_x = float(self.x_input.text())
#             self.sc.emg_y = float(self.y_input.text())
#         except:
#             pass
    
#     def closeEvent(self, event):
#         self.timer.stop()
#         event.accept()

class FittsTest(QWidget):
    def __init__(self, shared_context, dashboard):
        super().__init__()
        self.sc = shared_context
        self.dashboard = dashboard
        self.setWindowTitle("Fitts' Law Test")
        params = self.sc.params
        self.setFixedSize(*params['screen_size'])
        self.cursor_pos = [self.width() // 2, self.height() // 2]
        self.actual_velocity = [0.0, 0.0]
        self.hold_counter = 0
        self.targets_hit = 0
        self.targets_hit_current_combo = 0
        self.target_timer = 0
        self.ring_sequence = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        self.ring_index = 0
        self.combinations = [(r, t) for r in params.get('ring_radius_list', [300]) for t in params.get('target_radius_list', [20])]
        self.current_combo_idx = 0
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
        model_tag = str(self.sc.active_model_name).replace(" ", "_")
        _test_run = 'Test_' if self.dashboard.test_checkbox.isChecked() else ''
        filename = f"{fitts_folder}/{_test_run}Fitts_{ts}_{model_tag}.csv"
        self.log_file = open(filename, "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(["time", "frame", "mode", "model", "cursor_x", "cursor_y", "target_x", "target_y",
                             "radius", "X", "Y", "vx", "vy", "acc_x", "acc_y", "inside", "hold_count", "velocity",
                             "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"])
        self.frame_count = 0

    def init_target(self):
        params = self.sc.params
        mode = params['mode']
        w, h = self.width(), self.height()
        self.target_timer = 0
        if mode == "A":
            dist = random.randint(*params['target_distance_range'])
            angle = random.uniform(0, 2 * math.pi)
            cx, cy = self.cursor_pos
            radius = random.choice(params['target_radius_list'])
            self.target_radius = radius
            self.target_pos = [max(radius, min(w - radius, int(cx + math.cos(angle) * dist))),
                              max(radius, min(h - radius, int(cy + math.sin(angle) * dist)))]
        elif mode == "B":
            self.points = []
            cx, cy = self.width() // 2, self.height() // 2
            current_ring_radius, current_target_radius = self.combinations[self.current_combo_idx]
            self.target_radius = current_target_radius
            for i in range(12):
                angle = 2 * math.pi * i / 12
                self.points.append((int(cx + current_ring_radius * math.cos(angle)), 
                                  int(cy + current_ring_radius * math.sin(angle))))
            self.target_index = self.ring_sequence[self.ring_index % len(self.ring_sequence)]
            self.ring_index += 1
            self.target_pos = list(self.points[self.target_index])
        elif mode == "C":
            self.target_radius = random.choice(params['target_radius_list'])
            self.target_pos = [random.randint(self.target_radius, w - self.target_radius),
                              random.randint(self.target_radius, h - self.target_radius)]
            self.target_velocity = [random.choice([-1, 1]) * params['c_vel'], random.choice([-1, 1]) * params['c_vel']]

    def update_frame(self):
        params = self.sc.params
        x_in, y_in = self.sc.emg_x, self.sc.emg_y
        self.frame_count += 1
        self.target_timer += 1
        speed_mult = self.sc.speed_multiplier * VEL_CONSTANT
        desired_vx, desired_vy = x_in * speed_mult, y_in * speed_mult
        if params['physics']['enabled']:
            acc_x = (desired_vx - self.actual_velocity[0]) / params['physics']['mass']
            acc_y = (desired_vy - self.actual_velocity[1]) / params['physics']['mass']
            acc_x = max(-params['physics']['max_acceleration'], min(params['physics']['max_acceleration'], acc_x))
            acc_y = max(-params['physics']['max_acceleration'], min(params['physics']['max_acceleration'], acc_y))
            self.actual_velocity[0] = (self.actual_velocity[0] + acc_x) * params['physics']['damping']
            self.actual_velocity[1] = (self.actual_velocity[1] + acc_y) * params['physics']['damping']
        else:
            acc_x, acc_y = 0, 0
            self.actual_velocity = [desired_vx, desired_vy]
        self.cursor_pos[0] = max(0, min(self.cursor_pos[0] + self.actual_velocity[0], self.width()))
        self.cursor_pos[1] = max(0, min(self.cursor_pos[1] + self.actual_velocity[1], self.height()))
        dist = math.hypot(self.cursor_pos[0] - self.target_pos[0], self.cursor_pos[1] - self.target_pos[1])
        inside = dist <= self.target_radius
        self.hold_counter = self.hold_counter + 1 if inside else 0
        if params['mode'] in ["A", "B"]:
            if self.hold_counter >= params['hold_frames_required'] or self.target_timer >= params['target_timeout_frames']:
                self.targets_hit += 1
                self.targets_hit_current_combo += 1
                if self.target_timer >= params['target_timeout_frames'] and params['snap_back']:
                    self.cursor_pos[0] = self.target_pos[0]
                    self.cursor_pos[1] = self.target_pos[1]
                if params['mode'] == 'B':
                    if self.targets_hit_current_combo >= params['max_targets']:
                        self.current_combo_idx += 1
                        self.targets_hit_current_combo = 0
                        self.ring_index = 0
                        if self.current_combo_idx >= len(self.combinations): self.close()
                        else: self.init_target()
                    else:
                        self.init_target()
                else:
                    if self.targets_hit >= params['max_targets']: self.close()
                    else: self.init_target()
        elif params['mode'] == "C":
            for i in range(2):
                self.target_pos[i] += self.target_velocity[i]
                limit = self.width() if i == 0 else self.height()
                if self.target_pos[i] <= self.target_radius or self.target_pos[i] >= limit - self.target_radius:
                    self.target_velocity[i] *= -1
        self.logger.writerow([time.time(), self.frame_count, params['mode'], self.sc.active_model_name,
                             *self.cursor_pos, *self.target_pos, self.target_radius, x_in, y_in,
                             *self.actual_velocity, acc_x, acc_y, int(inside), self.hold_counter,
                             self.sc.raw_velocity, *self.sc.probs])
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#252525"))
        qp.setRenderHint(QPainter.Antialiasing)
        if self.sc.params['mode'] == "B":
            for x, y in self.points:
                qp.setPen(QPen(Qt.black, 2))
                qp.drawEllipse(int(x - self.target_radius), int(y - self.target_radius), 2 * self.target_radius, 2 * self.target_radius)
        qp.setBrush(QBrush(QColor("#10DA39") if self.hold_counter > 0 else QColor("#BD1B1B")))
        qp.drawEllipse(int(self.target_pos[0] - self.target_radius), int(self.target_pos[1] - self.target_radius), 2 * self.target_radius, 2 * self.target_radius)
        qp.setBrush(QBrush(QColor("#10C2DA")))
        qp.drawEllipse(int(self.cursor_pos[0]) - 5, int(self.cursor_pos[1]) - 5, 10, 10)

    def closeEvent(self, event):
        self.log_file.close()
        self.timer.stop()
        if self.dashboard: self.dashboard.test_window = None
        event.accept()