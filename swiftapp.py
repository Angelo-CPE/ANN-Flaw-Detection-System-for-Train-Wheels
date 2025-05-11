import sys
import time
import cv2
import torch
import numpy as np
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QGroupBox, QTextEdit,
                            QComboBox, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint

# ==============================================
# MODEL DEFINITION (Same as Before)
# ==============================================
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==============================================
# CAMERA THREAD WITH DUAL MODE FUNCTIONALITY
# ==============================================
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str)  # (mode, message)
    measurement_signal = pyqtSignal(float)
    fps_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._current_mode = "Flaw Detection"  # Default mode
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.measurement_points = []
        self.diameter_px = 0
        self.reference_length = 10.0  # cm (default reference)
        self.pixel_to_cm_ratio = 1.0

    def set_mode(self, mode):
        self._current_mode = mode
        self.status_signal.emit(mode, f"Switched to {mode} mode")

    def set_reference(self, length_cm):
        self.reference_length = length_cm
        if self.diameter_px > 0:
            self.pixel_to_cm_ratio = self.reference_length / self.diameter_px
            self.status_signal.emit("Diameter Measurement", 
                                  f"Calibrated: {self.reference_length}cm = {self.diameter_px}px")

    def add_measurement_point(self, point):
        if len(self.measurement_points) < 2:
            self.measurement_points.append(point)
            if len(self.measurement_points) == 2:
                self.calculate_diameter()

    def calculate_diameter(self):
        if len(self.measurement_points) == 2:
            p1, p2 = self.measurement_points
            self.diameter_px = np.sqrt((p2.x()-p1.x())**2 + (p2.y()-p1.y())**2)
            diameter_cm = self.diameter_px * self.pixel_to_cm_ratio
            self.measurement_signal.emit(diameter_cm)
            self.measurement_points = []  # Reset for next measurement

    def load_model(self):
        input_size = 2019
        self.model = ANNModel(input_size=input_size).to(self.device)
        model_path = 'D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/ANN_model.pth'  # Update this path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess_image(self, frame):
        frame_gray = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)
        hog_features = hog(frame_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        signal = np.mean(frame_gray, axis=0)
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.pad(np.diff(phase) / (2.0 * np.pi), (0, 1), mode='constant')
        combined_features = np.concatenate([hog_features, amplitude_envelope, frequency])
        if len(combined_features) != 2019:
            combined_features = np.resize(combined_features, 2019)
        return combined_features

    def run(self):
        # Try GStreamer first, fallback to default camera
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

        self.load_model()
        self.status_signal.emit(self._current_mode, "System Ready")

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Process frame based on current mode
                processed_frame = frame.copy()
                
                if self._current_mode == "Flaw Detection":
                    if self.frame_count % 3 == 0:  # Process every 3rd frame
                        features = self.preprocess_image(frame)
                        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(features_tensor)
                            _, predicted = torch.max(outputs, 1)
                        status = "Flaw Detected!" if predicted.item() == 1 else "No Flaw"
                        self.status_signal.emit(self._current_mode, status)
                        
                        # Add visual indicator
                        color = (0, 0, 255) if "Flaw" in status else (0, 255, 0)
                        cv2.putText(processed_frame, status, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        cv2.rectangle(processed_frame, (10, 10), (300, 80), color, 2)
                
                elif self._current_mode == "Diameter Measurement":
                    # Draw measurement points and line
                    if len(self.measurement_points) == 1:
                        cv2.circle(processed_frame, 
                                  (self.measurement_points[0].x(), self.measurement_points[0].y()), 
                                  5, (0, 255, 255), -1)
                    elif len(self.measurement_points) == 2:
                        p1 = (self.measurement_points[0].x(), self.measurement_points[0].y())
                        p2 = (self.measurement_points[1].x(), self.measurement_points[1].y())
                        cv2.line(processed_frame, p1, p2, (0, 255, 255), 2)
                        cv2.circle(processed_frame, p1, 5, (0, 255, 255), -1)
                        cv2.circle(processed_frame, p2, 5, (0, 255, 255), -1)
                    
                    # Add measurement guide text
                    cv2.putText(processed_frame, "Click two points to measure", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    fps = 10 / (time.time() - self.start_fps_time)
                    self.start_fps_time = time.time()
                    self.fps_signal.emit(fps)

                # Convert to QImage
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def gstreamer_pipeline(self):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )

# ==============================================
# MAIN APPLICATION WINDOW
# ==============================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Wheel Inspection System - Jetson Nano")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Layout
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Left Panel (Camera View)
        self.left_panel = QVBoxLayout()
        self.camera_label = ClickableLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.left_panel.addWidget(self.camera_label)
        
        # Right Panel (Controls)
        self.right_panel = QVBoxLayout()
        
        # Mode Selection
        self.mode_group = QGroupBox("Inspection Mode")
        self.mode_layout = QVBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Flaw Detection", "Diameter Measurement"])
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        
        # Reference Length Setup (for Diameter Measurement)
        self.reference_group = QGroupBox("Calibration")
        self.reference_layout = QVBoxLayout()
        self.reference_label = QLabel("Reference Length (cm):")
        self.reference_slider = QSlider(Qt.Horizontal)
        self.reference_slider.setRange(1, 50)
        self.reference_slider.setValue(10)
        self.reference_value = QLabel("10.0 cm")
        self.reference_slider.valueChanged.connect(self.update_reference)
        
        # Status Panel
        self.status_group = QGroupBox("System Status")
        self.status_layout = QVBoxLayout()
        self.mode_status = QLabel("Mode: Flaw Detection")
        self.detection_status = QLabel("Status: Initializing...")
        self.measurement_status = QLabel("Diameter: -- cm")
        self.fps_status = QLabel("FPS: --")
        
        # Log Console
        self.log_group = QGroupBox("Event Log")
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        
        # Control Buttons
        self.control_group = QGroupBox("Controls")
        self.control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.exit_btn = QPushButton("Exit")
        
        # Setup UI
        self.setup_ui()
        
        # Camera Thread
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.measurement_signal.connect(self.update_measurement)
        self.camera_thread.fps_signal.connect(self.update_fps)
        self.camera_label.clicked.connect(self.handle_click)

    def setup_ui(self):
        # Set fonts
        font_large = QFont()
        font_large.setPointSize(12)
        font_large.setBold(True)
        
        font_small = QFont()
        font_small.setPointSize(10)
        
        # Mode Selection
        self.mode_selector.setFont(font_large)
        self.mode_layout.addWidget(self.mode_selector)
        self.mode_group.setLayout(self.mode_layout)
        
        # Reference Setup
        self.reference_label.setFont(font_small)
        self.reference_value.setFont(font_large)
        self.reference_layout.addWidget(self.reference_label)
        self.reference_layout.addWidget(self.reference_slider)
        self.reference_layout.addWidget(self.reference_value)
        self.reference_group.setLayout(self.reference_layout)
        
        # Status Panel
        self.mode_status.setFont(font_large)
        self.detection_status.setFont(font_large)
        self.measurement_status.setFont(font_large)
        self.fps_status.setFont(font_small)
        self.status_layout.addWidget(self.mode_status)
        self.status_layout.addWidget(self.detection_status)
        self.status_layout.addWidget(self.measurement_status)
        self.status_layout.addWidget(self.fps_status)
        self.status_group.setLayout(self.status_layout)
        
        # Log Console
        self.log_console.setFont(font_small)
        self.log_group.setLayout(QVBoxLayout())
        self.log_group.layout().addWidget(self.log_console)
        
        # Control Buttons
        self.start_btn.setFont(font_large)
        self.stop_btn.setFont(font_large)
        self.exit_btn.setFont(font_large)
        self.control_layout.addWidget(self.start_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.control_layout.addWidget(self.exit_btn)
        self.control_group.setLayout(self.control_layout)
        
        # Assemble Right Panel
        self.right_panel.addWidget(self.mode_group)
        self.right_panel.addWidget(self.reference_group)
        self.right_panel.addWidget(self.status_group)
        self.right_panel.addWidget(self.log_group)
        self.right_panel.addWidget(self.control_group)
        self.right_panel.addStretch()
        
        # Combine Main Layout
        self.main_layout.addLayout(self.left_panel, 70)
        self.main_layout.addLayout(self.right_panel, 30)
        
        # Button Connections
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.exit_btn.clicked.connect(self.close_app)
        
        # Apply Styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #aaa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 3px;
                color: #7d7;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QTextEdit {
                background-color: #252525;
                color: #e0e0e0;
                border: 1px solid #444;
            }
            QComboBox {
                background-color: #444;
                color: white;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #444;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -5px 0;
                background: #7d7;
                border-radius: 9px;
            }
        """)
        
        # Initially hide reference controls
        self.reference_group.hide()
        self.measurement_status.hide()

    def change_mode(self, mode):
        self.camera_thread.set_mode(mode)
        self.mode_status.setText(f"Mode: {mode}")
        
        if mode == "Diameter Measurement":
            self.reference_group.show()
            self.measurement_status.show()
            self.detection_status.hide()
        else:
            self.reference_group.hide()
            self.measurement_status.hide()
            self.detection_status.show()

    def update_reference(self, value):
        self.reference_value.setText(f"{value}.0 cm")
        self.camera_thread.set_reference(float(value))

    def handle_click(self, pos):
        if self.camera_thread._current_mode == "Diameter Measurement":
            self.camera_thread.add_measurement_point(pos)
            self.log_console.append(f"[{time.strftime('%H:%M:%S')}] Measurement point at ({pos.x()}, {pos.y()})")

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, mode, message):
        if mode == "Flaw Detection":
            self.detection_status.setText(f"Status: {message}")
            if "Flaw" in message:
                self.detection_status.setStyleSheet("color: #f55;")
                self.log_console.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ {message}")
            else:
                self.detection_status.setStyleSheet("color: #7d7;")
        else:
            self.log_console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def update_measurement(self, diameter):
        self.measurement_status.setText(f"Diameter: {diameter:.2f} cm")
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] Measured diameter: {diameter:.2f} cm")

    def update_fps(self, fps):
        self.fps_status.setText(f"FPS: {fps:.1f}")

    def start_camera(self):
        self.camera_thread.start()
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] System started")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_camera(self):
        self.camera_thread.stop()
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] System stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def close_app(self):
        self.camera_thread.stop()
        self.close()

# Clickable QLabel for measurement points
class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)
    
    def mousePressEvent(self, event):
        self.clicked.emit(event.pos())
        super().mousePressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())