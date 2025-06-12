import sys
import os
import time
import cv2
import base64
import torch
import numpy as np
import requests
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QFrame, QMessageBox, QSizePolicy,
                            QGridLayout, QStackedWidget, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QFontDatabase, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QPropertyAnimation, QEasingCurve

# Import VL53L0X library
try:
    import VL53L0X
except ImportError:
    print("VL53L0X library not found. Distance measurement will be simulated.")
    VL53L0X = None

def send_report_to_backend(status, recommendation, image_base64, name=None, trainNumber=None, compartmentNumber=None, wheelNumber=None, wheel_diameter=None):
    # Validate status before sending
    if status not in ["FLAW DETECTED", "NO FLAW"]:
        print("Invalid status, defaulting to 'NO FLAW'")
        status = "NO FLAW"
        recommendation = "For Constant Monitoring"

    import tempfile
    backend_url = "https://ann-flaw-detection-system-for-train.onrender.com/api/reports"  # Updated endpoint

    try:
        # Create a temporary image file from base64
        img_data = base64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_data)
            temp_img_path = temp_img.name

        with open(temp_img_path, 'rb') as img_file:
            files = {
                'image': ('inspection.jpg', img_file, 'image/jpeg')
            }
            data = {
                'status': status,
                'recommendation': recommendation,
                'name': name,
                'trainNumber': str(trainNumber),
                'compartmentNumber': str(compartmentNumber),
                'wheelNumber': str(wheelNumber),
                'wheel_diameter': str(wheel_diameter)
            }

            # Add timeout and better error handling
            try:
                response = requests.post(backend_url, files=files, data=data, timeout=10)
                
                if response.status_code == 201:
                    print("Report sent successfully!")
                    return True
                else:
                    print(f"Failed to send report. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"Network error while sending report: {e}")
                return False
                
    except Exception as e:
        print(f"Error preparing report: {e}")
        return False
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

class DistanceSensorThread(QThread):
    distance_measured = pyqtSignal(int)
    measurement_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, tof):
        super().__init__()
        self._run_flag = True
        self.tof = tof

    def run(self):
        try:
            start_time = time.time()
            while time.time() - start_time < 1.0 and self._run_flag:
                distance = self.tof.get_distance()
                if distance > 0:
                    self.distance_measured.emit(distance)
                time.sleep(0.05)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                self.tof.stop_ranging()
                self.tof.close()
            except Exception as e:
                print(f"Sensor cleanup error: {e}")
            self.measurement_complete.emit()

    def stop(self):
        self._run_flag = False
        self.wait()

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str)
    test_complete_signal = pyqtSignal(np.ndarray, str, str)
    animation_signal = pyqtSignal()
    enable_buttons_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._testing = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_frame = None
        self.load_model()

    def load_model(self):
        try:
            input_size = 2020
            self.model = ANNModel(input_size=input_size).to(self.device)
            model_path = 'ANN_model.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                print("ANN model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def unload_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print("Model unloaded successfully")

    def preprocess_image(self, frame):
        try:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2GRAY)
            hog_features = hog(frame_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            signal = np.mean(frame_gray, axis=0)
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            combined_features = np.concatenate([hog_features, amplitude_envelope])
            if len(combined_features) != 2020:
                combined_features = np.resize(combined_features, 2020)
            return combined_features
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return np.zeros(2019)

    def start_test(self):
        if self.model is None:
            self.status_signal.emit("Error", "Model not loaded")
            return
            
        self._testing = True
        self.enable_buttons_signal.emit(False)
        self.process_captured_image()

    def process_captured_image(self):
        if self.last_frame is None or self.model is None:
            self.status_signal.emit("Error", "No frame captured or model not loaded")
            self.enable_buttons_signal.emit(True)
            return

        try:
            cv2.imwrite("captured_frame.jpg", self.last_frame)
            features = self.preprocess_image(self.last_frame)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(features_tensor)
                print("Model Raw Output:", outputs.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                print("Probabilities:", probs.cpu().numpy())
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs, 1)
            
            if predicted.item() == 1:
                status = "FLAW DETECTED"
                recommendation = "For Repair/Replacement"
            else:
                status = "NO FLAW"
                recommendation = "For Constant Monitoring"
            
            self.status_signal.emit(status, recommendation)
            self.test_complete_signal.emit(self.last_frame, status, recommendation)
            self.animation_signal.emit()
        except Exception as e:
            print(f"Error processing image: {e}")
            self.status_signal.emit("Error", "Processing failed")
        finally:
            self.enable_buttons_signal.emit(True)

    def run(self):
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, "
            "format=NV12, framerate=20/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=480, height=360, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )

        cap = None
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("GStreamer camera failed. Restarting nvargus-daemon...")
                os.system("sudo systemctl restart nvargus-daemon")
                time.sleep(1.0)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                print("Falling back to USB camera...")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 20)

            if not cap.isOpened():
                print("Error: Cannot access any camera")
                return

            print("Camera Ready")

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    self.last_frame = frame.copy()
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image)

        except Exception as e:
            print(f"Camera error: {e}")
            print("Camera error")
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)
        
        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(120, Qt.SmoothTransformation))
        self.layout.addWidget(self.logo_label)
        
        # Title
        self.title_label = QLabel("Wheel Inspection System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 28px;
                color: #333;
                padding: 10px 0;
            }
        """)
        self.layout.addWidget(self.title_label)
        
        # Buttons
        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(20)
        
        # 1. Inspection Button
        self.inspection_btn = QPushButton("INSPECTION")
        self.inspection_btn.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
        """)
        self.inspection_btn.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))
        self.button_layout.addWidget(self.inspection_btn)
        
        # 2. Calibration Button
        self.calibration_btn = QPushButton("CALIBRATION")
        self.calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #111;
            }
            QPushButton:pressed {
                background-color: #000;
            }
        """)
        self.calibration_btn.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(3))
        self.button_layout.addWidget(self.calibration_btn)
        
        self.layout.addLayout(self.button_layout)
        self.layout.addStretch(1)
        self.setLayout(self.layout)

class SelectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #333;
                border: none;
                padding: 5px;
                font-family: 'Montserrat';
                font-size: 14px;
                text-align: left;
            }
            QPushButton:hover {
                color: #e60000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(0))
        self.layout.addWidget(self.back_button)
        
        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(100, Qt.SmoothTransformation))
        self.layout.addWidget(self.logo_label)
        
        # Title
        self.title_label = QLabel("Wheel Inspection")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 24px;
                color: #333;
                padding: 10px 0;
            }
        """)
        self.layout.addWidget(self.title_label)
        
        # Train Selection
        self.train_layout = QVBoxLayout()
        self.train_layout.setSpacing(5)
        self.train_label = QLabel("Select Train Number")
        self.train_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 16px;
                color: #555;
            }
        """)
        self.train_layout.addWidget(self.train_label)
        
        self.train_slider = QSlider(Qt.Horizontal)
        self.train_slider.setRange(1, 20)
        self.train_slider.setValue(1)
        self.train_slider.setStyleSheet("""
            QSlider {
                height: 40px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #ddd;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 24px;
                height: 24px;
                margin: -8px 0;
                background: #e60000;
                border-radius: 12px;
            }
        """)
        self.train_layout.addWidget(self.train_slider)
        
        self.train_value = QLabel("1")
        self.train_value.setAlignment(Qt.AlignCenter)
        self.train_value.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 20px;
                color: #333;
            }
        """)
        self.train_layout.addWidget(self.train_value)
        self.layout.addLayout(self.train_layout)
        
        # Compartment Selection
        self.compartment_layout = QVBoxLayout()
        self.compartment_layout.setSpacing(5)
        self.compartment_label = QLabel("Select Compartment Number")
        self.compartment_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 16px;
                color: #555;
            }
        """)
        self.compartment_layout.addWidget(self.compartment_label)
        
        self.compartment_slider = QSlider(Qt.Horizontal)
        self.compartment_slider.setRange(1, 8)
        self.compartment_slider.setValue(1)
        self.compartment_slider.setStyleSheet(self.train_slider.styleSheet())
        self.compartment_layout.addWidget(self.compartment_slider)
        
        self.compartment_value = QLabel("1")
        self.compartment_value.setAlignment(Qt.AlignCenter)
        self.compartment_value.setStyleSheet(self.train_value.styleSheet())
        self.compartment_layout.addWidget(self.compartment_value)
        self.layout.addLayout(self.compartment_layout)
        
        # Wheel Selection
        self.wheel_layout = QVBoxLayout()
        self.wheel_layout.setSpacing(5)
        self.wheel_label = QLabel("Select Wheel Number")
        self.wheel_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 16px;
                color: #555;
            }
        """)
        self.wheel_layout.addWidget(self.wheel_label)
        
        self.wheel_slider = QSlider(Qt.Horizontal)
        self.wheel_slider.setRange(1, 8)
        self.wheel_slider.setValue(1)
        self.wheel_slider.setStyleSheet(self.train_slider.styleSheet())
        self.wheel_layout.addWidget(self.wheel_slider)
        
        self.wheel_value = QLabel("1")
        self.wheel_value.setAlignment(Qt.AlignCenter)
        self.wheel_value.setStyleSheet(self.train_value.styleSheet())
        self.wheel_layout.addWidget(self.wheel_value)
        self.layout.addLayout(self.wheel_layout)
        
        # Start Button
        self.start_button = QPushButton("START INSPECTION")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
        """)
        self.start_button.clicked.connect(self.start_inspection)
        self.layout.addWidget(self.start_button)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        
        # Connect signals
        self.train_slider.valueChanged.connect(lambda: self.train_value.setText(str(self.train_slider.value())))
        self.compartment_slider.valueChanged.connect(lambda: self.compartment_value.setText(str(self.compartment_slider.value())))
        self.wheel_slider.valueChanged.connect(lambda: self.wheel_value.setText(str(self.wheel_slider.value())))

    def start_inspection(self):
        self.parent.trainNumber = self.train_slider.value()
        self.parent.compartmentNumber = self.compartment_slider.value()
        self.parent.wheelNumber = self.wheel_slider.value()
        self.parent.stacked_widget.setCurrentIndex(2)

class InspectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.setup_animations()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #333;
                border: none;
                padding: 5px;
                font-family: 'Montserrat';
                font-size: 14px;
                text-align: left;
            }
            QPushButton:hover {
                color: #e60000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))
        self.layout.addWidget(self.back_button)
        
        # Camera Panel
        self.camera_panel = QFrame()
        self.camera_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("QLabel { background: black; border: none; }")
        
        self.camera_layout.addWidget(self.camera_label)
        self.camera_panel.setLayout(self.camera_layout)
        self.layout.addWidget(self.camera_panel, 60)
        
        # Control Panel
        self.control_panel = QFrame()
        self.control_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.control_layout = QVBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(10)
        
        # Current Selection
        self.selection_label = QLabel()
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 16px;
                color: #333;
                padding: 5px;
            }
        """)
        self.update_selection_label()
        self.control_layout.addWidget(self.selection_label)
        
        # Status Panel
        self.status_panel = QFrame()
        self.status_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.status_layout = QVBoxLayout()
        self.status_layout.setContentsMargins(5, 5, 5, 5)
        self.status_layout.setSpacing(5)
        
        self.status_title = QLabel("INSPECTION STATUS")
        self.status_title.setAlignment(Qt.AlignCenter)
        self.status_title.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                padding-bottom: 2px;
                border-bottom: 1px solid #eee;
            }
        """)
        
        self.status_indicator = QLabel("READY")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 16px;
                padding-top: 2px;
                padding-bottom: 0px;
            }
        """)

        self.recommendation_indicator = QLabel()
        self.recommendation_indicator.setAlignment(Qt.AlignCenter)
        self.recommendation_indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat';
                font-size: 14px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
        """)

        self.diameter_label = QLabel("Wheel Diameter: -")
        self.diameter_label.setAlignment(Qt.AlignCenter)
        self.diameter_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-family: 'Montserrat';
                font-size: 14px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
        """)
        self.diameter_label.hide()
        
        self.status_layout.addWidget(self.status_title)
        self.status_layout.addWidget(self.status_indicator)
        self.status_layout.addWidget(self.recommendation_indicator)
        self.status_layout.addWidget(self.diameter_label)
        self.status_panel.setLayout(self.status_layout)
        self.control_layout.addWidget(self.status_panel)
        
        # Button Panel
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(10, 10, 10, 10)
        self.button_layout.setSpacing(10)
        
        # 1. Detect Flaws Button
        self.detect_btn = QPushButton("DETECT FLAWS")
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #cc0000; }
            QPushButton:pressed { background-color: #b30000; }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """)
        
        # 2. Measure Diameter Button
        self.measure_btn = QPushButton("MEASURE DIAMETER")
        self.measure_btn.setEnabled(False)
        self.measure_btn.setCursor(Qt.PointingHandCursor)
        self.measure_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
            QPushButton:hover { background-color: #111; }
            QPushButton:pressed { background-color: #000; }
        """)
        
        # 3. Save Report Button
        self.save_btn = QPushButton("SAVE REPORT")
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #006600;
                color: white;
                border: none;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
            QPushButton:hover { background-color: #004400; }
            QPushButton:pressed { background-color: #002200; }
        """)
        
        # 4. Reset Button
        self.reset_btn = QPushButton("NEW INSPECTION")
        self.reset_btn.setVisible(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #222; }
            QPushButton:pressed { background-color: #111; }
        """)
        
        self.button_layout.addWidget(self.detect_btn)
        self.button_layout.addWidget(self.measure_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_panel.setLayout(self.button_layout)
        self.control_layout.addWidget(self.button_panel)
        
        self.control_panel.setLayout(self.control_layout)
        self.layout.addWidget(self.control_panel, 40)
        self.setLayout(self.layout)
        
        # Connect signals
        self.detect_btn.clicked.connect(self.parent.detect_flaws)
        self.measure_btn.clicked.connect(self.parent.measure_diameter)
        self.save_btn.clicked.connect(self.parent.save_report)
        self.reset_btn.clicked.connect(self.parent.reset_ui)

    def update_selection_label(self):
        self.selection_label.setText(
            f"Train: {self.parent.trainNumber} | "
            f"Compartment: {self.parent.compartmentNumber} | "
            f"Wheel: {self.parent.wheelNumber}"
        )

    def setup_animations(self):
        self.status_animation = QPropertyAnimation(self.status_indicator, b"windowOpacity")
        self.status_animation.setDuration(300)
        self.status_animation.setStartValue(0.7)
        self.status_animation.setEndValue(1.0)

class CalibrationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #333;
                border: none;
                padding: 5px;
                font-family: 'Montserrat';
                font-size: 14px;
                text-align: left;
            }
            QPushButton:hover {
                color: #e60000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(0))
        self.layout.addWidget(self.back_button)
        
        # Title
        self.title_label = QLabel("Calibration")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 28px;
                color: #333;
                padding: 10px 0;
            }
        """)
        self.layout.addWidget(self.title_label)
        
        # Placeholder content
        self.placeholder_label = QLabel("Wheel Diameter Measurement Calibration")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 18px;
                color: #666;
                padding: 20px 0;
            }
        """)
        self.layout.addWidget(self.placeholder_label)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel Inspection")
        self.setWindowIcon(QIcon("logo.png"))
        self.showFullScreen()
            
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.current_distance = 680
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background: white;")
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.central_widget.setLayout(self.main_layout)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.home_page = HomePage(self)
        self.selection_page = SelectionPage(self)
        self.inspection_page = InspectionPage(self)
        self.calibration_page = CalibrationPage(self)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home_page)          # Index 0
        self.stacked_widget.addWidget(self.selection_page)     # Index 1
        self.stacked_widget.addWidget(self.inspection_page)   # Index 2
        self.stacked_widget.addWidget(self.calibration_page)  # Index 3
        
        # Setup camera thread
        self.setup_camera_thread()
        
        # Connect signals
        self.inspection_page.reset_btn.clicked.connect(self.reset_ui)

    def setup_camera_thread(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.test_complete_signal.connect(self.handle_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_animation)
        self.camera_thread.enable_buttons_signal.connect(self.set_buttons_enabled)
        self.camera_thread.start()

    def update_image(self, qt_image):
        self.inspection_page.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.inspection_page.camera_label.width(), 
            self.inspection_page.camera_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

    def update_status(self, status, recommendation):
        if status in ["FLAW DETECTED", "NO FLAW"]:
            if hasattr(self, 'current_distance') and self.current_distance != 680:
                self.inspection_page.diameter_label.setText(f"Wheel Diameter: {self.current_distance} mm")
            else:
                self.inspection_page.diameter_label.setText("Wheel Diameter: Measure Next")
            self.inspection_page.diameter_label.show()
        else:
            self.inspection_page.diameter_label.hide()
            
        self.inspection_page.status_indicator.setText(status)
        self.inspection_page.recommendation_indicator.setText(recommendation)
        
        if status == "FLAW DETECTED":
            self.inspection_page.status_indicator.setStyleSheet("""
                QLabel {
                    color: red;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 10px 0;
                }
            """)
            self.inspection_page.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 3px solid red;
                }
            """)
        elif status == "NO FLAW":
            self.inspection_page.status_indicator.setStyleSheet("""
                QLabel {
                    color: #00CC00;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 10px 0;
                }
            """)
            self.inspection_page.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 3px solid #00CC00;
                }
            """)
        else:
            self.inspection_page.status_indicator.setStyleSheet("""
                QLabel {
                    color: black;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 10px 0;
                }
            """)
            self.inspection_page.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: none;
                }
            """)
        
        self.trigger_animation()

    def detect_flaws(self):
        self.inspection_page.status_indicator.setText("ANALYZING...")
        self.inspection_page.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 10px 0;
            }
        """)
        self.inspection_page.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.inspection_page.diameter_label.hide()
        
        # Immediately disable the button for visual feedback
        self.inspection_page.detect_btn.setEnabled(False)
        self.inspection_page.measure_btn.setEnabled(False)
        self.inspection_page.save_btn.setEnabled(False)
        
        self.camera_thread.start_test()

    def measure_diameter(self):
        self.inspection_page.diameter_label.setText("Measuring...")
        self.inspection_page.diameter_label.show()

        self.inspection_page.detect_btn.setEnabled(False)
        self.inspection_page.measure_btn.setEnabled(False)
        self.inspection_page.save_btn.setEnabled(False)

        try:
            self.tof = VL53L0X.VL53L0X()
            self.tof.open()
            time.sleep(0.1)
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)

            self.sensor_thread = DistanceSensorThread(self.tof)
            self.sensor_thread.distance_measured.connect(self.update_distance)
            self.sensor_thread.measurement_complete.connect(self.on_measurement_complete)
            self.sensor_thread.error_occurred.connect(self.on_measurement_error)
            self.sensor_thread.start()

        except Exception as e:
            print(f"Sensor init error: {e}")
            self.on_measurement_error("Sensor init error")

    def update_distance(self, distance):
        self.current_distance = distance
        self.inspection_page.diameter_label.setText(f"Wheel Diameter: {distance} mm")
        self.inspection_page.detect_btn.setVisible(False)
        self.inspection_page.measure_btn.setVisible(False)
        self.inspection_page.reset_btn.setVisible(True)
        self.inspection_page.save_btn.setVisible(True)
        self.inspection_page.diameter_label.show()

    def on_measurement_error(self, error_msg):
        print(f"Measurement error: {error_msg}")
        self.on_measurement_complete()

    def on_measurement_complete(self):
        # After measurement, show Reset and Save buttons
        self.inspection_page.detect_btn.setVisible(False)
        self.inspection_page.measure_btn.setVisible(False)
        self.inspection_page.save_btn.setEnabled(True)
        self.inspection_page.save_btn.setVisible(True)
        self.inspection_page.reset_btn.setVisible(True)

    def handle_test_complete(self, image, status, recommendation):
        # Ensure we have a valid image
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            print("Error: Invalid image received from test")
            self.test_image = None
        else:
            self.test_image = image.copy()  # Make a copy to ensure we don't lose it
            
        self.test_status = status
        self.test_recommendation = recommendation

        # After detection, show:
        # - Detect Flaws (disabled)
        # - Measure Diameter (enabled)
        self.inspection_page.detect_btn.setEnabled(False)
        self.inspection_page.detect_btn.setVisible(True)
        self.inspection_page.measure_btn.setEnabled(True)
        self.inspection_page.measure_btn.setVisible(True)
        self.inspection_page.save_btn.setEnabled(False)
        self.inspection_page.save_btn.setVisible(False)
        self.inspection_page.reset_btn.setVisible(False)

    def set_buttons_enabled(self, enabled):
        # Only enable measure button if we have a test result
        if hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"]:
            self.inspection_page.measure_btn.setEnabled(enabled)
        else:
            self.inspection_page.measure_btn.setEnabled(False)
        
        # Only enable save button if we have both test result and measurement
        if (hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"] and self.current_distance != 680):
            self.inspection_page.save_btn.setEnabled(enabled)
        else:
            self.inspection_page.save_btn.setEnabled(False)

    def trigger_animation(self):
        self.inspection_page.status_animation.start()

    def save_report(self):
        msg = QMessageBox()
        msg.setWindowTitle("Save Report")
        msg.setText("Save this inspection report?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border: 1px solid #ddd;
                font-family: 'Montserrat';
            }
            QLabel {
                color: black;
                font-size: 16px;
            }
            QPushButton {
                background-color: #006600;
                color: white;
                border: none;
                padding: 10px 20px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                min-width: 100px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #004400; }
            #qt_msgbox_buttonbox { border-top: 1px solid #ddd; padding-top: 20px; }
        """)
        
        if msg.exec_() == QMessageBox.Save:
            # Check if test_image exists and is valid
            if self.test_image is None or not isinstance(self.test_image, np.ndarray) or self.test_image.size == 0:
                QMessageBox.critical(self, "Error", "No valid inspection image available to save.")
                return
                
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            try:
                # Convert image to base64
                success, buffer = cv2.imencode('.jpg', self.test_image)
                if not success:
                    raise ValueError("Failed to encode image")
                    
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                report_name = f"Train {self.trainNumber} - Compartment {self.compartmentNumber} - Wheel {self.wheelNumber}"
                
                # Send report and check if it was successful
                success = send_report_to_backend(
                    status=self.test_status,
                    recommendation=self.test_recommendation,
                    image_base64=image_base64,
                    name=report_name,
                    trainNumber=self.trainNumber,
                    compartmentNumber=self.compartmentNumber,
                    wheelNumber=self.wheelNumber,
                    wheel_diameter=self.current_distance
                )
                
                if success:
                    # Only reset if save was successful
                    self.reset_ui()
                    QMessageBox.information(self, "Success", "Report saved successfully!")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to save report. Please check your connection and try again.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

    def reset_ui(self):
        self.stacked_widget.setCurrentIndex(1)  # Go back to selection page
        self.inspection_page.status_indicator.setText("READY")
        self.inspection_page.recommendation_indicator.setText("")
        self.inspection_page.diameter_label.setText("Wheel Diameter: -")
        self.inspection_page.diameter_label.hide()
        self.inspection_page.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 10px 0;
            }
        """)
        self.inspection_page.recommendation_indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        self.inspection_page.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        
        # Reset buttons to initial state
        self.inspection_page.detect_btn.setEnabled(True)
        self.inspection_page.detect_btn.setVisible(True)
        self.inspection_page.measure_btn.setEnabled(False)
        self.inspection_page.measure_btn.setVisible(True)
        self.inspection_page.save_btn.setEnabled(False)
        self.inspection_page.save_btn.setVisible(False)
        self.inspection_page.reset_btn.setVisible(False)

        # Reset data
        self.current_distance = 680
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None
        
        # Reload the model for next use
        self.camera_thread.load_model()

    def closeEvent(self, event):
        self.camera_thread.stop()
        if hasattr(self, 'sensor_thread'):
            self.sensor_thread.stop()
        event.accept()

if __name__ == "__main__":
    os.environ["QT_QUICK_BACKEND"] = "software"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load Montserrat font if available
    font_db = QFontDatabase()
    if "Montserrat" not in font_db.families():
        # Try to load the font from file if not found
        font_paths = [
            "Montserrat-Regular.ttf",
            "fonts/Montserrat-Regular.ttf",
            "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_id = font_db.addApplicationFont(path)
                if font_id != -1:
                    break
    
    # Set application font
    font = QFont("Montserrat", 12)
    app.setFont(font)
    
    palette = app.palette()
    palette.setColor(palette.Window, QColor(255, 255, 255))
    palette.setColor(palette.WindowText, QColor(0, 0, 0))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = App()
    window.show()
    
    try:
        os.nice(10)
    except:
        pass
    
    sys.exit(app.exec_())