import sys
import csv
import math
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

class FittsReplay(QWidget):
    def __init__(self, csv_file, w=1690, h=980, fps=60):
        super().__init__()
        self.setWindowTitle("Fitts' Law Replay")
        self.setFixedSize(w, h)
        
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
                
        self.frame_idx = 0
        self.max_frames = len(self.data)
        
        if self.max_frames == 0:
            print("[ERROR] Empty CSV.")
            sys.exit(1)
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // fps)
        self.show()

    def update_frame(self):
        if self.frame_idx < self.max_frames - 1:
            self.frame_idx += 1
            self.update()
        else:
            self.timer.stop()

    def paintEvent(self, event):
        row = self.data[self.frame_idx]
        
        mode = row['mode']
        cx = float(row['cursor_x'])
        cy = float(row['cursor_y'])
        tx = float(row['target_x'])
        ty = float(row['target_y'])
        tr = float(row['radius'])
        hold = int(row['hold_count'])
        
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#252525"))
        qp.setRenderHint(QPainter.Antialiasing)
        
        if mode == "B":
            center_x, center_y = self.width() // 2, self.height() // 2
            ring_r = math.hypot(tx - center_x, ty - center_y)
            for i in range(12):
                angle = 2 * math.pi * i / 12
                px = int(center_x + ring_r * math.cos(angle))
                py = int(center_y + ring_r * math.sin(angle))
                qp.setPen(QPen(Qt.black, 2))
                qp.setBrush(Qt.NoBrush)
                qp.drawEllipse(int(px - tr), int(py - tr), int(2 * tr), int(2 * tr))
                
        qp.setBrush(QBrush(QColor("#10DA39") if hold > 0 else QColor("#BD1B1B")))
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(int(tx - tr), int(ty - tr), int(2 * tr), int(2 * tr))
        
        qp.setBrush(QBrush(QColor("#10C2DA")))
        qp.drawEllipse(int(cx) - 5, int(cy) - 5, 10, 10)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python replay.py <path_to_log.csv>")
    #     sys.exit(1)
        
    app = QApplication(sys.argv)
    window = FittsReplay(r'fitts_logs\amir2\Test_Fitts_2026-04-07_12-11-18_mlp_within_raw_2.csv')
    sys.exit(app.exec())