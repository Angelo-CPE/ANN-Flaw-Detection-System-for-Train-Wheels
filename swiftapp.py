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
                            QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QFontDatabase, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QPropertyAnimation, QEasingCurve

# VL53L0X import with error handling
VL53L0X_AVAILABLE = False
try:
    import VL53L0X
    VL53L0X_AVAILABLE = True
except ImportError:
    print("VL53L0X library not found. Distance measurement will be simulated.")
    VL53L0X = None
except Exception as e:
    print(f"Error importing VL53L0X: {e}. Distance measurement will be simulated.")
    VL53L0X = None

def send_report_to_backend(status, recommendation, image_base64, name=None, notes="", trainNumber=None, compartmentNumber=None, wheelNumber=None, wheel_diameter=None):
    backend_url = "http://localhost:5000/api/reports"  # Update as needed

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "recommendation": recommendation,
        "image_path": image_base64,
        "name": name,
        "notes": notes,
        "trainNumber": trainNumber,
        "compartmentNumber": compartmentNumber,
        "wheelNumber": wheelNumber,
        "wheel_diameter": wheel_diameter
    }

    try:
        response = requests.post(backend_url, json=report, timeout=5)
        if response.status_code == 201:
            print("Report sent successfully!")
        else:
            print(f"Failed to send report: {response.text}")
    except Exception as e:
        print(f"Error sending report: {e}")

class DistanceSensorThread(QThread):
    distance_measured = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.tof = None

    def run(self):
        try:
            import VL53L0X
            self.tof = VL53L0X.VL53L0X()
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)
            time.sleep(1)
            distance = self.tof.get_distance()
            self.tof.stop_ranging()
            self.tof.close()
            self.distance_measured.emit(distance)
        except Exception as e:
            print(f"VL53L0X error: {e}")
            self.distance_measured.emit(680)

    distance_measured = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tof = None
        self.sensor_initialized = False

    def initialize_sensor(self):
        if not VL53L0X_AVAILABLE:
            return False
            
        try:
            self.tof = VL53L0X.VL53L0X()
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)
            self.sensor_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing distance sensor: {e}")
            self.sensor_initialized = False
            return False

    def run(self):
        if not self.initialize_sensor():
            # Simulate distance measurement if sensor is not available
            while self._run_flag:
                simulated_distance = 680  # Default simulated value
                self.distance_measured.emit(simulated_distance)
                time.sleep(0.2)
            return

        try:
            while self._run_flag:
                distance = self.tof.get_distance()
                if distance > 0:  # Only emit valid measurements
                    self.distance_measured.emit(distance)
                time.sleep(0.2)  # Delay to avoid spamming measurements

        except Exception as e:
            print(f"Error with distance sensor: {e}")
            # Emit simulated value if there's an error
            self.distance_measured.emit(680)
        finally:
            if self.tof is not None:
                try:
                    self.tof.stop_ranging()
                    self.tof.close()
                except:
                    pass

    def stop(self):
        self._run_flag = False
        self.wait()

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
        self._countdown = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_frame = None
        self.load_model()
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)

    def load_model(self):
        try:
            input_size = 2019
            self.model = ANNModel(input_size=input_size).to(self.device)
            model_path = 'ANN_model.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, frame):
        try:
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
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return np.zeros(2019)

    def start_test(self):
        self.diameter_btn.setEnabled(False)
                
        self.status_indicator.setText("ANALYZING...")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 15px 0;
            }
        """)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.diameter_label.hide()

        # Run ToF sensor one-time read
        self.sensor_thread = DistanceSensorThread()
        self.sensor_thread.distance_measured.connect(self.update_distance)
        self.sensor_thread.finished.connect(self.camera_thread.start_test)
        self.sensor_thread.start()

        if self.model is None:
            self.status_signal.emit("Error", "Model not loaded")
            return
            
        self._testing = True
        self._countdown = 0.5
        self.enable_buttons_signal.emit(False)
        self.countdown_timer.start(500)

    
    def update_countdown(self):
        self._countdown = -1
        self.countdown_timer.stop()
        self.process_captured_image()

    def process_captured_image(self):
        if self.last_frame is None or self.model is None:
            self.status_signal.emit("Error", "No frame captured or model not loaded")
            self.enable_buttons_signal.emit(True)
            return

        try:
            features = self.preprocess_image(self.last_frame)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
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
        cap = None
        try:
            pipeline = self.gstreamer_pipeline()
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("Failed to open with GStreamer, trying default camera")
                cap = cv2.VideoCapture(0)
                
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

            self.status_signal.emit("Ready", "")

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    self.last_frame = frame.copy()
                    processed_frame = frame.copy()
                    
                    if self._testing and self._countdown >= 0:
                        cv2.putText(processed_frame, "IMAGE CAPTURED", 
                                   (processed_frame.shape[1]//2 - 150, processed_frame.shape[0]//2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image)

        except Exception as e:
            print(f"Camera error: {e}")
            self.status_signal.emit("Error", "Camera error")
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self._run_flag = False
        self.countdown_timer.stop()
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

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel Inspection")
        self.setWindowIcon(QIcon("icon.png"))
        self.setFixedSize(800, 480)
            
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.current_distance = 0
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background: white;")
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.central_widget.setLayout(self.main_layout)
        
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Camera Panel
        self.camera_panel = QFrame()
        self.camera_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet("QLabel { background: black; border: none; }")
        
        self.camera_layout.addWidget(self.camera_label)
        self.camera_panel.setLayout(self.camera_layout)
        
        # Control Panel
        self.control_panel = QFrame()
        self.control_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.control_layout = QVBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(0)
        
        # Status Panel
        self.status_panel = QFrame()
        self.status_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.status_layout = QVBoxLayout()
        self.status_layout.setContentsMargins(10, 10, 10, 10)
        
        # Logo
        self.logo_space = QLabel()
        self.logo_space.setAlignment(Qt.AlignCenter)
        self.logo_space.setFixedHeight(80)
        self.logo_space.setStyleSheet("background: transparent;")
        
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_space.setPixmap(logo_pixmap.scaledToHeight(150, Qt.SmoothTransformation))
        
        # Number controls
        self.number_controls = QFrame()
        self.number_controls.setStyleSheet("QFrame { background: transparent; }")
        self.number_layout = QGridLayout()
        self.number_layout.setContentsMargins(0, 0, 0, 0)
        self.number_layout.setSpacing(5)
        
        # Train Number
        self.train_label = QLabel("Train")
        self.train_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 15px;
                color: #333;
                padding-right: 6px;
            }
        """)
        
        self.train_decrement = QPushButton("-")
        self.train_decrement.setFixedSize(25, 25)
        self.train_decrement.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        self.trainNumber_label = QLabel("1")
        self.trainNumber_label.setAlignment(Qt.AlignCenter)
        self.trainNumber_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Black';
                font-size: 16px;
                color: #111;
                min-width: 32px;
                min-height: 25px;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
            }
        """)
        self.train_increment = QPushButton("+")
        self.train_increment.setFixedSize(25, 25)
        self.train_increment.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        # Compartment Number
        self.compartment_label = QLabel("Compartment")
        self.compartment_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 15px;
                color: #333;
                padding-right: 6px;
            }
        """)
        
        self.compartment_decrement = QPushButton("-")
        self.compartment_decrement.setFixedSize(25, 25)
        self.compartment_decrement.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        self.compartmentNumber_label = QLabel("1")
        self.compartmentNumber_label.setAlignment(Qt.AlignCenter)
        self.compartmentNumber_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Black';
                font-size: 16px;
                color: #111;
                min-width: 32px;
                min-height: 25px;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
            }
        """)
        
        self.compartment_increment = QPushButton("+")
        self.compartment_increment.setFixedSize(25, 25)
        self.compartment_increment.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        # Wheel Number
        self.wheel_label = QLabel("Wheel")
        self.wheel_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 15px;
                color: #333;
                padding-right: 6px;
            }
        """)
        
        self.wheel_decrement = QPushButton("-")
        self.wheel_decrement.setFixedSize(25, 25)
        self.wheel_decrement.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        self.wheelNumber_label = QLabel("1")
        self.wheelNumber_label.setAlignment(Qt.AlignCenter)
        self.wheelNumber_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Black';
                font-size: 16px;
                color: #111;
                min-width: 32px;
                min-height: 25px;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
            }
        """)
        
        self.wheel_increment = QPushButton("+")
        self.wheel_increment.setFixedSize(25, 25)
        self.wheel_increment.setStyleSheet("""
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #bbb;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                min-width: 25px;
                max-width: 25px;
                height: 25px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        
        # Add to layout
        self.number_layout.addWidget(self.train_label, 0, 0)
        self.number_layout.addWidget(self.train_decrement, 0, 1)
        self.number_layout.addWidget(self.trainNumber_label, 0, 2)
        self.number_layout.addWidget(self.train_increment, 0, 3)
        
        self.number_layout.addWidget(self.compartment_label, 1, 0)
        self.number_layout.addWidget(self.compartment_decrement, 1, 1)
        self.number_layout.addWidget(self.compartmentNumber_label, 1, 2)
        self.number_layout.addWidget(self.compartment_increment, 1, 3)
        
        self.number_layout.addWidget(self.wheel_label, 2, 0)
        self.number_layout.addWidget(self.wheel_decrement, 2, 1)
        self.number_layout.addWidget(self.wheelNumber_label, 2, 2)
        self.number_layout.addWidget(self.wheel_increment, 2, 3)
        
        self.number_controls.setLayout(self.number_layout)
        
        self.status_title = QLabel("INSPECTION STATUS")
        self.status_title.setAlignment(Qt.AlignCenter)
        self.status_title.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat Black';
                font-size: 20px;
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
                font-size: 15px;
                padding-top: 2px;
                padding-bottom: 0px;
            }
        """)

        self.analyzing_label = QLabel("ANALYZING")
        self.analyzing_label.setAlignment(Qt.AlignCenter)
        self.analyzing_label.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 15px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
        """)
        self.analyzing_label.hide()

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
        
        self.status_layout.addWidget(self.logo_space)
        self.status_layout.addWidget(self.number_controls)
        self.status_layout.addWidget(self.status_title)
        self.status_layout.addWidget(self.status_indicator)
        self.status_layout.addWidget(self.analyzing_label)
        self.status_layout.addWidget(self.recommendation_indicator)
        self.status_layout.addWidget(self.diameter_label)
        self.status_panel.setLayout(self.status_layout)
        
        # Button Panel
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(20, 20, 20, 20)
        
        self.start_btn = QPushButton("START INSPECTION")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #cc0000; }
            QPushButton:pressed { background-color: #b30000; }
        """)

        self.diameter_btn = QPushButton("MEASURE DIAMETER")
        self.diameter_btn.setCursor(Qt.PointingHandCursor)
        self.diameter_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
            QPushButton:hover { background-color: #111; }
            QPushButton:pressed { background-color: #000; }
        """)
        self.diameter_btn = QPushButton("MEASURE DIAMETER")
        self.diameter_btn.setCursor(Qt.PointingHandCursor)
        self.diameter_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #111; }
            QPushButton:pressed { background-color: #000; }
        """)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #cc0000; }
            QPushButton:pressed { background-color: #990000; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        
        self.save_btn = QPushButton("SAVE RESULTS")
        self.save_btn.setEnabled(False)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: black;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #333; }
            QPushButton:pressed { background-color: #000; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        
        self.button_layout.addWidget(self.start_btn)
        self.button_layout.addWidget(self.diameter_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_panel.setLayout(self.button_layout)
        
        self.control_layout.addWidget(self.status_panel)
        self.control_layout.addWidget(self.button_panel)
        self.control_panel.setLayout(self.control_layout)
        
        self.content_layout.addWidget(self.camera_panel, 60)
        self.content_layout.addWidget(self.control_panel, 40)
        
        self.main_layout.addLayout(self.content_layout)
        
        self.setup_animations()
        self.setup_camera_thread()
        # self.setup_distance_sensor()
        self.connect_signals()
        self.setup_number_controls()

    def setup_number_controls(self):
        self.train_decrement.clicked.connect(lambda: self.update_number('train', -1))
        self.train_increment.clicked.connect(lambda: self.update_number('train', 1))
        self.compartment_decrement.clicked.connect(lambda: self.update_number('compartment', -1))
        self.compartment_increment.clicked.connect(lambda: self.update_number('compartment', 1))
        self.wheel_decrement.clicked.connect(lambda: self.update_number('wheel', -1))
        self.wheel_increment.clicked.connect(lambda: self.update_number('wheel', 1))

    def update_number(self, number_type, change):
        if number_type == 'train':
            self.trainNumber = max(1, min(20, self.trainNumber + change))
            self.trainNumber_label.setText(str(self.trainNumber))
        elif number_type == 'compartment':
            self.compartmentNumber = max(1, min(8, self.compartmentNumber + change))
            self.compartmentNumber_label.setText(str(self.compartmentNumber))
        elif number_type == 'wheel':
            self.wheelNumber = max(1, min(8, self.wheelNumber + change))
            self.wheelNumber_label.setText(str(self.wheelNumber))

    def setup_animations(self):
        self.status_animation = QPropertyAnimation(self.status_indicator, b"windowOpacity")
        self.status_animation.setDuration(300)
        self.status_animation.setStartValue(0.7)
        self.status_animation.setEndValue(1.0)

    def setup_camera_thread(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.test_complete_signal.connect(self.handle_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_animation)
        self.camera_thread.enable_buttons_signal.connect(self.set_buttons_enabled)
        self.camera_thread.start()

    def setup_distance_sensor(self):
        self.distance_sensor_thread = DistanceSensorThread()
        self.distance_sensor_thread.distance_measured.connect(self.update_distance)
        self.distance_sensor_thread.start()
        QTimer.singleShot(1000, self.distance_sensor_thread.stop)

    def update_distance(self, distance):
        self.current_distance = distance
        self.diameter_label.setText(f"Wheel Diameter: {distance} mm")
        self.diameter_label.show()
        if hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"]:
            self.save_btn.setEnabled(True)
            self.current_distance = distance
        if hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"]:
            self.save_btn.setEnabled(True)
            self.diameter_label.setText(f"Wheel Diameter: {distance} mm")
            self.diameter_label.show()
            self.current_distance = distance

    def connect_signals(self):
        self.diameter_btn.clicked.connect(self.measure_diameter)
        self.start_btn.clicked.connect(self.start_test)

    def set_buttons_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled and self.save_btn.isEnabled())

    def trigger_animation(self):
        self.status_animation.start()

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, status, recommendation):
        if status in ["FLAW DETECTED", "NO FLAW"]:
            self.diameter_label.setText(f"Wheel Diameter: {self.current_distance} mm")
            self.diameter_label.show()
        else:
            self.diameter_label.hide()
            
        self.status_indicator.setText(status)
        self.recommendation_indicator.setText(recommendation)
        
        if status == "FLAW DETECTED":
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: red;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 15px 0;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    color: black;
                    font-family: 'Montserrat';
                    font-size: 14px;
                    padding: 10px 0;
                }
            """)
            self.diameter_label.setStyleSheet("""
                QLabel {
                    color: #333;
                    font-family: 'Montserrat';
                    font-size: 16px;
                    padding: 5px 0;
                }
            """)
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 4px solid red;
                }
            """)
        elif status == "NO FLAW":
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #00CC00;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 15px 0;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    color: black;
                    font-family: 'Montserrat';
                    font-size: 14px;
                    padding: 10px 0;
                }
            """)
            self.diameter_label.setStyleSheet("""
                QLabel {
                    color: #333;
                    font-family: 'Montserrat';
                    font-size: 16px;
                    padding: 5px 0;
                }
            """)
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 4px solid #00CC00;
                }
            """)
        else:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: black;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 15px 0;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-family: 'Montserrat';
                    font-size: 14px;
                    padding: 10px 0;
                }
            """)
            self.diameter_label.setStyleSheet("""
                QLabel {
                    color: #333;
                    font-family: 'Montserrat';
                    font-size: 16px;
                    padding: 5px 0;
                }
            """)
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: none;
                }
            """)
        
        self.trigger_animation()

    def start_test(self):
        self.diameter_btn.setEnabled(False)
                
        self.status_indicator.setText("ANALYZING...")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 15px 0;
            }
        """)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.diameter_label.hide()

        # Run ToF sensor one-time read
        self.camera_thread.start_test()

        self.status_indicator.setText("ANALYZING...")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 15px 0;
            }
        """)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.diameter_label.hide()
        self.camera_thread.start_test()

    
    def measure_diameter(self):
        
        self.distance_sensor_thread = DistanceSensorThread()
        self.distance_sensor_thread.distance_measured.connect(self.update_distance)
        self.distance_sensor_thread.start()
        QTimer.singleShot(1000, self.distance_sensor_thread.stop)
    
    def reset_app(self):
        self.start_btn.setText("START INSPECTION")
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status_indicator.setText("READY")
        self.recommendation_indicator.setText("")
        self.diameter_label.setText("Wheel Diameter: -")
        self.diameter_label.hide()
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 15px 0;
            }
        """)
        self.recommendation_indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.start_btn.disconnect()
        self.start_btn.clicked.connect(self.start_test)

    def handle_test_complete(self, image, status, recommendation):
        self.test_image = image
        self.test_status = status
        self.test_recommendation = recommendation
        self.test_status = status
        self.test_recommendation = recommendation
        self.diameter_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.start_btn.setText("RESET")
        self.start_btn.disconnect()
        self.start_btn.clicked.connect(self.reset_app)
        self.save_btn.clicked.connect(self.save_results)

    def save_results(self):
        msg = QMessageBox()
        msg.setWindowTitle("Save Results")
        msg.setText("Save this inspection result?")
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
                font-size: 14px;
            }
            QPushButton {
                background-color: red;
                color: white;
                border: none;
                padding: 8px 16px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #cc0000; }
            #qt_msgbox_buttonbox { border-top: 1px solid #ddd; padding-top: 16px; }
        """)
        
        if msg.exec_() == QMessageBox.Save:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', self.test_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            report_name = f"Train {self.trainNumber} - Compartment {self.compartmentNumber} - Wheel {self.wheelNumber}"
            
            send_report_to_backend(
                status=self.test_status,
                recommendation=self.test_recommendation,
                image_base64=image_base64,
                name=report_name,
                trainNumber=self.trainNumber,
                compartmentNumber=self.compartmentNumber,
                wheelNumber=self.wheelNumber,
                wheel_diameter=self.current_distance
            )
        self.save_btn.setEnabled(False)

    def closeEvent(self, event):
        self.camera_thread.stop()
        if hasattr(self, 'distance_sensor_thread'): self.distance_sensor_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = app.palette()
    palette.setColor(palette.Window, QColor(255, 255, 255))
    palette.setColor(palette.WindowText, QColor(0, 0, 0))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(palette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(palette.Text, QColor(0, 0, 0))
    palette.setColor(palette.Button, QColor(240, 240, 240))
    palette.setColor(palette.ButtonText, QColor(0, 0, 0))
    palette.setColor(palette.BrightText, QColor(255, 255, 255))
    palette.setColor(palette.Highlight, QColor(255, 0, 0))
    palette.setColor(palette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = App()
    window.show()
    sys.exit(app.exec_())
