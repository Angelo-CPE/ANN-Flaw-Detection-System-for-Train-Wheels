import sys
import os
import time
import serial
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
from simulated_ina219 import INA219

def send_report_to_backend(status, recommendation, image_base64, name=None, trainNumber=None, compartmentNumber=None, wheelNumber=None, wheel_diameter=None):
    if status not in ["FLAW DETECTED", "NO FLAW"]:
        print("Invalid status, defaulting to 'NO FLAW'")
        status = "NO FLAW"
        recommendation = "For Constant Monitoring"

    import tempfile
    backend_url = "https://ann-flaw-detection-system-for-train.onrender.com/api/reports"

    try:
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

    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        super().__init__()
        self._run_flag = True
        self.port = port
        self.baudrate = baudrate
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def run(self):
        retry_count = 0
        while retry_count < self.max_retries and self._run_flag:
            try:
                with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                    print(f"Connected to {self.port} at {self.baudrate} baud")
                    
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                    
                    time.sleep(2)
                    
                    while ser.in_waiting:
                        ser.readline()

                    ser.write(b'R\n')
                    
                    start_time = time.time()
                    while time.time() - start_time < 5 and self._run_flag:
                        if ser.in_waiting:
                            line = ser.readline().decode('ascii', errors='ignore').strip()
                            if line:
                                print(f"Received from Arduino: {line}")
                                try:
                                    distance = int(line)
                                    if distance > 0:
                                        self.distance_measured.emit(distance)
                                        self.measurement_complete.emit()
                                        return
                                    elif distance == 0:
                                        self.error_occurred.emit("No new distance measurement available from Arduino.")
                                        break
                                    elif distance == -1:
                                        self.error_occurred.emit("Arduino: Sensor initialization error.")
                                        break
                                    elif distance == -2:
                                        self.error_occurred.emit("Arduino: Sensor timeout occurred.")
                                        break
                                except ValueError:
                                    print(f"Non-integer data received from Arduino: {line}")
                                    continue
                    
                    self.error_occurred.emit("No valid distance measurement received from Arduino within timeout.")
                    break
                
            except serial.SerialException as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    print(f"Serial port error (attempt {retry_count}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    self.error_occurred.emit(f"Serial port error: {str(e)}. Make sure Arduino is connected and the correct port is selected.")
            except Exception as e:
                self.error_occurred.emit(f"Unexpected error in DistanceSensorThread: {str(e)}")
                break
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

class BatteryMonitorThread(QThread):
    battery_percentage_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.ina219 = INA219()
        self.last_voltage = 0
        self.smoothing_factor = 0.2  # Smoothing factor for exponential moving average

    def run(self):
        while self._run_flag:
            try:
                bus_voltage = self.ina219.getBusVoltage_V()
                # Simple exponential smoothing
                self.last_voltage = (self.smoothing_factor * bus_voltage) + ((1 - self.smoothing_factor) * self.last_voltage)
                
                # More accurate battery percentage calculation
                # Assuming 8.4V is 100% and 6.0V is 0% (adjust these values based on your battery specs)
                min_voltage = 6.0
                max_voltage = 8.4
                
                # Clamp voltage within range
                clamped_voltage = max(min_voltage, min(max_voltage, self.last_voltage))
                
                # Calculate percentage
                percentage = ((clamped_voltage - min_voltage) / (max_voltage - min_voltage)) * 100
                percentage = max(0, min(100, int(percentage)))
                
                self.battery_percentage_signal.emit(percentage)
            except Exception as e:
                print(f"Error reading battery data: {e}")
                self.battery_percentage_signal.emit(-1)
            time.sleep(5)

    def stop(self):
        self._run_flag = False
        self.wait()

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
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background: white;")
        
        # Main vertical layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Camera Panel (top section)
        self.camera_panel = QFrame()
        self.camera_panel.setStyleSheet("QFrame { background: black; border: none; }")
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_layout = QVBoxLayout(self.camera_panel)
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_layout.setSpacing(0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("QLabel { background: black; border: none; }")
        
        # Add widgets to camera layout
        self.camera_layout.addWidget(self.camera_label)
        
        # Control Panel (bottom section)
        self.control_panel = QFrame()
        self.control_panel.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-top: 2px solid #ddd;
            }
        """)
        self.control_panel.setFixedHeight(400)  # Fixed height for control panel
        self.control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(20, 10, 20, 10)
        self.control_layout.setSpacing(15)
        
        # Status Panel
        self.status_panel = QFrame()
        self.status_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.status_layout = QVBoxLayout(self.status_panel)
        self.status_layout.setContentsMargins(5, 5, 5, 5)
        self.status_layout.setSpacing(10)
        
        # Logo
        self.logo_space = QLabel()
        self.logo_space.setAlignment(Qt.AlignCenter)
        self.logo_space.setFixedHeight(60)
        self.logo_space.setStyleSheet("background: transparent;")
        
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_space.setPixmap(logo_pixmap.scaledToHeight(60, Qt.SmoothTransformation))
        
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
        
        # Battery Percentage Label
        self.battery_label = QLabel("Battery: --%")
        self.battery_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.battery_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        self.status_layout.addWidget(self.battery_label)
        self.status_layout.addWidget(self.diameter_label)
        self.status_panel.setLayout(self.status_layout)
        
        # Button Panel
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(10, 10, 10, 10)
        self.button_layout.setSpacing(8)
        
        # 1. Detect Flaws Button
        self.detect_btn = QPushButton("DETECT FLAWS")
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """)
        
        # 2. Measure Diameter Button
        self.measure_btn = QPushButton("MEASURE DIAMETER")
        self.measure_btn.setEnabled(True)
        self.measure_btn.setCursor(Qt.PointingHandCursor)
        self.measure_btn.setStyleSheet("""
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
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
            QPushButton:hover { background-color: #004400; }
            QPushButton:pressed { background-color: #002200; }
        """)
        
        # 4. Reset Button
        self.reset_btn = QPushButton("RESET")
        self.reset_btn.setVisible(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #222; }
            QPushButton:pressed { background-color: #111; }
        """)
        
        self.button_layout.addWidget(self.detect_btn)
        self.button_layout.addWidget(self.measure_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_layout.addStretch(1)
        
        self.button_panel.setLayout(self.button_layout)
        
        self.control_layout.addWidget(self.status_panel)
        self.control_layout.addWidget(self.button_panel)
        self.control_panel.setLayout(self.control_layout)
        
        # Add panels to main layout with proper stretch factors
        self.main_layout.addWidget(self.camera_panel)
        self.main_layout.addWidget(self.control_panel)  
        
        self.setup_animations()
        self.setup_camera_thread()
        self.connect_signals()
        self.setup_number_controls()

        # Battery Monitor Thread
        self.battery_monitor_thread = BatteryMonitorThread()
        self.battery_monitor_thread.battery_percentage_signal.connect(self.update_battery_percentage)
        self.battery_monitor_thread.start()

    def setup_animations(self):
        """Initialize animations for the UI elements"""
        self.status_animation = QPropertyAnimation(self.status_indicator, b"windowOpacity")
        self.status_animation.setDuration(300)
        self.status_animation.setStartValue(0.7)
        self.status_animation.setEndValue(1.0)

    def setup_number_controls(self):
        self.train_decrement.clicked.connect(self.decrement_train_number)
        self.train_increment.clicked.connect(self.increment_train_number)
        self.compartment_decrement.clicked.connect(self.decrement_compartment_number)
        self.compartment_increment.clicked.connect(self.increment_compartment_number)
        self.wheel_decrement.clicked.connect(self.decrement_wheel_number)
        self.wheel_increment.clicked.connect(self.increment_wheel_number)

    def decrement_train_number(self):
        if self.trainNumber > 1:
            self.trainNumber -= 1
            self.trainNumber_label.setText(str(self.trainNumber))

    def increment_train_number(self):
        self.trainNumber += 1
        self.trainNumber_label.setText(str(self.trainNumber))

    def decrement_compartment_number(self):
        if self.compartmentNumber > 1:
            self.compartmentNumber -= 1
            self.compartmentNumber_label.setText(str(self.compartmentNumber))

    def increment_compartment_number(self):
        self.compartmentNumber += 1
        self.compartmentNumber_label.setText(str(self.compartmentNumber))

    def decrement_wheel_number(self):
        if self.wheelNumber > 1:
            self.wheelNumber -= 1
            self.wheelNumber_label.setText(str(self.wheelNumber))

    def increment_wheel_number(self):
        self.wheelNumber += 1
        self.wheelNumber_label.setText(str(self.wheelNumber))

    def setup_camera_thread(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.test_complete_signal.connect(self.on_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_animation)
        self.camera_thread.enable_buttons_signal.connect(self.enable_buttons)
        self.camera_thread.start()

    def connect_signals(self):
        self.detect_btn.clicked.connect(self.camera_thread.start_test)
        self.measure_btn.clicked.connect(self.start_distance_measurement)
        self.save_btn.clicked.connect(self.save_report)
        self.reset_btn.clicked.connect(self.reset_application)

    def start_distance_measurement(self):
        self.diameter_label.hide()
        self.status_indicator.setText("MEASURING...")
        self.recommendation_indicator.setText("")
        self.enable_buttons(False)
        self.distance_thread = DistanceSensorThread()
        self.distance_thread.distance_measured.connect(self.on_distance_measured)
        self.distance_thread.measurement_complete.connect(self.on_measurement_complete)
        self.distance_thread.error_occurred.connect(self.on_distance_error)
        self.distance_thread.start()

    def on_distance_measured(self, distance):
        self.current_distance = distance
        if self.current_distance > 0:
            # More accurate diameter calculation based on sensor geometry
            # Assuming sensor is mounted at a known distance from wheel center
            # This is just an example - adjust the formula based on your actual setup
            sensor_to_wheel_center = 500  # mm (adjust this based on your mounting)
            diameter = 2 * (sensor_to_wheel_center - self.current_distance)
            self.diameter_label.setText(f"Wheel Diameter: {diameter:.1f} mm")
            self.diameter_label.show()
        else:
            self.diameter_label.setText("Wheel Diameter: N/A")
            self.diameter_label.show()

    def on_measurement_complete(self):
        self.status_indicator.setText("READY")
        self.enable_buttons(True)

    def on_distance_error(self, message):
        self.status_indicator.setText("ERROR")
        self.recommendation_indicator.setText(message)
        self.enable_buttons(True)
        self.diameter_label.hide()
        QMessageBox.warning(self, "Measurement Error", message)

    def update_image(self, qt_image):
        """Maintain aspect ratio while scaling the camera image"""
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(
            self.camera_label.width(), 
            self.camera_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)
        self.camera_label.setAlignment(Qt.AlignCenter)

    def update_status(self, status, recommendation):
        self.status_indicator.setText(status)
        self.recommendation_indicator.setText(recommendation)
        if status == "FLAW DETECTED":
            self.status_indicator.setStyleSheet("QLabel { color: red; font-family: 'Montserrat ExtraBold'; font-size: 15px; padding-top: 2px; padding-bottom: 0px; }")
            self.save_btn.setVisible(True)
            self.reset_btn.setVisible(True)
        else:
            self.status_indicator.setStyleSheet("QLabel { color: green; font-family: 'Montserrat ExtraBold'; font-size: 15px; padding-top: 2px; padding-bottom: 0px; }")
            self.save_btn.setVisible(True)
            self.reset_btn.setVisible(True)

    def on_test_complete(self, frame, status, recommendation):
        self.test_image = frame
        self.test_status = status
        self.test_recommendation = recommendation

    def trigger_animation(self):
        self.analyzing_label.show()
        self.animation = QPropertyAnimation(self.analyzing_label, b"geometry")
        self.animation.setDuration(1000)
        
        current_rect = self.analyzing_label.geometry()
        self.animation.setStartValue(current_rect)
        
        end_rect = current_rect.translated(10, 0)
        self.animation.setEndValue(end_rect)
        
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.setLoopCount(2)
        self.animation.finished.connect(self.hide_analyzing_label)
        self.animation.start()

    def hide_analyzing_label(self):
        self.analyzing_label.hide()

    def enable_buttons(self, enable):
        self.detect_btn.setEnabled(enable)
        self.measure_btn.setEnabled(enable)

    def save_report(self):
        if self.test_image is not None and self.test_status is not None and self.test_recommendation is not None:
            _, buffer = cv2.imencode('.jpg', self.test_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            success = send_report_to_backend(
                self.test_status,
                self.test_recommendation,
                image_base64,
                name="Wheel Inspection Report",
                trainNumber=self.trainNumber,
                compartmentNumber=self.compartmentNumber,
                wheelNumber=self.wheelNumber,
                wheel_diameter=self.diameter_label.text().replace("Wheel Diameter: ", "").replace(" mm", "")
            )
            if success:
                QMessageBox.information(self, "Report", "Report sent successfully!")
            else:
                QMessageBox.warning(self, "Report", "Failed to send report.")
        else:
            QMessageBox.warning(self, "Report", "No test data to save.")

    def reset_application(self):
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.trainNumber_label.setText(str(self.trainNumber))
        self.compartmentNumber_label.setText(str(self.compartmentNumber))
        self.wheelNumber_label.setText(str(self.wheelNumber))
        self.status_indicator.setText("READY")
        self.status_indicator.setStyleSheet("QLabel { color: black; font-family: 'Montserrat ExtraBold'; font-size: 15px; padding-top: 2px; padding-bottom: 0px; }")
        self.recommendation_indicator.setText("")
        self.diameter_label.hide()
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.enable_buttons(True)

    def update_battery_percentage(self, percentage):
        if percentage == -1:
            self.battery_label.setText("Battery: Error")
        else:
            self.battery_label.setText(f"Battery: {percentage}%")

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.battery_monitor_thread.stop()
        event.accept()

if __name__ == '__main__':
    try:
        import serial
    except ImportError:
        print("Python 'serial' module not found. Please install it using: pip install pyserial")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    # Load custom fonts
    font_paths = [
        "Montserrat-Black.ttf",
        "Montserrat-Bold.ttf",
        "Montserrat-ExtraBold.ttf",
        "Montserrat-SemiBold.ttf",
        "Montserrat-Regular.ttf"
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            QFontDatabase.addApplicationFont(font_path)
        else:
            print(f"Font file not found: {font_path}")

    ex = App()
    ex.show()
    sys.exit(app.exec_())