import sys
import os
import time
import cv2
import base64
import torch
import numpy as np
import requests
import smbus
import serial
from serial.tools import list_ports
from serial import SerialException
from ina219 import INA219
from ina219 import DeviceRangeError
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QFrame, QMessageBox, QSizePolicy,
                            QGridLayout, QStackedWidget, QSlider, QStyle, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QFontDatabase, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QPropertyAnimation, QEasingCurve
import serial

def send_report_to_backend(status, recommendation, image_base64, name=None, trainNumber=None, compartmentNumber=None, wheelNumber=None, wheel_diameter=None, token=None):
    # Validate status before sending
    if status not in ["FLAW DETECTED", "NO FLAW"]:
        print("Invalid status, defaulting to 'NO FLAW'")
        status = "NO FLAW"
        recommendation = "For Constant Monitoring"

    import tempfile
    backend_url = "https://ann-flaw-detection-system-for-train.onrender.com/api/reports"

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
            headers = {
                'Authorization': f'Bearer {token}'
            } if token else {}

            try:
                response = requests.post(
                    backend_url, 
                    files=files, 
                    data=data, 
                    headers=headers,
                    timeout=10
                )
                
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

class BatteryMonitorThread(QThread):
    battery_updated = pyqtSignal(float, float)  # voltage, percentage
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.ina = None
        
    def run(self):
        try:
            # Jetson Nano uses I2C bus 1 (like Raspberry Pi models 2 and later)
            bus_number = 1
            
            # Initialize INA219 with correct parameters for Jetson Nano
            self.ina = INA219(
                shunt_ohms=0.1,
                max_expected_amps=0.6,
                address=0x42,
                busnum=bus_number
            )
            self.ina.configure(voltage_range=self.ina.RANGE_16V)
            
            print(f"Battery monitor initialized successfully on bus {bus_number}")
        except Exception as e:
            print(f"Battery monitor initialization failed: {e}")
            self.battery_updated.emit(0, 0)
            return
        
        while self._run_flag:
            try:
                voltage = self.ina.voltage()
                # Calculate percentage for 2-cell Li-ion battery
                # (6.0V = 0%, 8.4V = 100%)
                percentage = min(100, max(0, (voltage - 6.0) / (8.4 - 6.0) * 100))
                self.battery_updated.emit(voltage, percentage)
            except DeviceRangeError as e:
                print(f"Battery read error: {e}")
            except Exception as e:
                print(f"Battery monitor error: {e}")
            
            time.sleep(5)  # Update every 5 seconds
    
    def stop(self):
        self._run_flag = False
        self.wait(2000)

class BatteryIndicator(QWidget):
    def __init__(self, parent=None, compact=False):
        super().__init__(parent)
        self.compact = compact
        self.voltage = 0.0
        self.percentage = 0
        self.setStyleSheet("background: transparent;")
        
        # Set size based on whether it's compact mode
        if compact:
            self.setFixedSize(50, 25)
        else:
            self.setFixedSize(80, 40)
        
    def update_battery(self, voltage, percentage):
        self.voltage = voltage
        self.percentage = percentage
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw battery outline
        body_width = w * 0.7
        body_height = h * 0.6
        tip_width = w * 0.1
        tip_height = h * 0.3
        
        body_x = (w - body_width - tip_width) / 2
        body_y = (h - body_height) / 2
        tip_x = body_x + body_width
        tip_y = body_y + (body_height - tip_height) / 2
        
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(Qt.transparent)
        painter.drawRoundedRect(body_x, body_y, body_width, body_height, 2, 2)
        painter.drawRect(tip_x, tip_y, tip_width, tip_height)  # Battery tip
        
        # Calculate fill width based on percentage
        fill_width = max(0, min(body_width - 4, (body_width - 4) * self.percentage / 100))
        
        # Choose color based on battery level
        if self.percentage > 60:
            color = QColor(0, 200, 0)  # Green
        elif self.percentage > 20:
            color = QColor(255, 165, 0)  # Orange
        else:
            color = QColor(220, 0, 0)  # Red
        
        # Draw battery fill
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(body_x+2, body_y+2, fill_width, body_height-4, 1, 1)
        
        # Draw percentage text only in non-compact mode
        if not self.compact:
            painter.setPen(Qt.black)
            font = QFont("Arial", 8)
            painter.setFont(font)
            painter.drawText(0, 0, w, h, Qt.AlignCenter, f"{int(self.percentage)}%")

class CalibrationSerialThread(QThread):
    distance_measured = pyqtSignal(float)  # Raw distance in mm
    measurement_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        super().__init__()
        self._run_flag = True
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None

    def run(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            valid_readings = []
            start_time = time.time()
            
            # Collect readings for 2 seconds
            while time.time() - start_time < 2.0 and self._run_flag:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    try:
                        distance = float(line)
                        if distance > 0:  # Ignore error values
                            valid_readings.append(distance)
                            self.distance_measured.emit(distance)
                    except ValueError:
                        pass
                time.sleep(0.01)
                
            # Calculate median of valid readings
            if valid_readings:
                median_distance = float(np.median(valid_readings))
                self.distance_measured.emit(median_distance)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.measurement_complete.emit()

    def stop(self):
        self._run_flag = False
        self.wait()

class SerialReaderThread(QThread):
    diameter_measured = pyqtSignal(float)
    measurement_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    # Constants from Arduino code
    CHORD_L = 0.250  # metres (exact pad spacing)
    LEVER_GAIN = 3.00  # 3× mechanical amplifier
    LIFT_OFF_MM = 38.0  # sensor→lever gap when off-wheel
    
    # Calibration constants (update these with your actual calibration values)
    CAL_700_RAW = 200.0  # gap on 700 mm ring (bigger gap)
    CAL_625_RAW = 100.0  # gap on 625 mm ring (smaller gap)
    
    # Calculated constants
    M_SLOPE = (700.0 - 625.0) / (CAL_700_RAW - CAL_625_RAW)
    B_OFFS = 700.0 - M_SLOPE * CAL_700_RAW

    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        super().__init__()
        self._run_flag = True
        self.port       = port
        self.baudrate   = baudrate
        self.serial_conn = None

        # ←── NEW: how long to collect raw readings
        self.collection_time = 5.0  
        
        self.load_calibration_values()
    
    def load_calibration_values(self):
        try:
            if os.path.exists("calibration_values.txt"):
                with open("calibration_values.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(':')
                        if len(parts) < 2:
                            continue
                            
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        try:
                            if key == "700mm":
                                self.CAL_700_RAW = float(value)
                            elif key == "625mm":
                                self.CAL_625_RAW = float(value)
                            elif key == "M_SLOPE":
                                self.M_SLOPE = float(value)
                            elif key == "B_OFFS":
                                self.B_OFFS = float(value)
                        except ValueError:
                            print(f"Skipping invalid calibration value: {value}")
                            
                # Recalculate if values are missing
                if hasattr(self, 'CAL_700_RAW') and hasattr(self, 'CAL_625_RAW'):
                    self.M_SLOPE = (700.0 - 625.0) / (self.CAL_700_RAW - self.CAL_625_RAW)
                    self.B_OFFS = 700.0 - self.M_SLOPE * self.CAL_700_RAW
                else:
                    # Fallback to defaults
                    self.CAL_700_RAW = 200.0
                    self.CAL_625_RAW = 100.0
                    self.M_SLOPE = (700.0 - 625.0) / (self.CAL_700_RAW - self.CAL_625_RAW)
                    self.B_OFFS = 700.0 - self.M_SLOPE * self.CAL_700_RAW
                    
                print("Loaded calibration values:")
                print(f"CAL_700_RAW: {self.CAL_700_RAW}")
                print(f"CAL_625_RAW: {self.CAL_625_RAW}")
                print(f"M_SLOPE: {self.M_SLOPE}")
                print(f"B_OFFS: {self.B_OFFS}")
        except Exception as e:
            print(f"Error loading calibration values: {e}")
            # Fall back to defaults
            self.CAL_700_RAW = 200.0
            self.CAL_625_RAW = 100.0
            self.M_SLOPE = (700.0 - 625.0) / (self.CAL_700_RAW - self.CAL_625_RAW)
            self.B_OFFS = 700.0 - self.M_SLOPE * self.CAL_700_RAW

    def calculate_diameter(self, raw_mm):
        dia1, raw1 = 700.0, self.CAL_700_RAW
        dia2, raw2 = 625.0, self.CAL_625_RAW
        slope = (dia1 - dia2) / (raw1 - raw2)
        offset = dia1 - slope * raw1
        raw_dia = slope * raw_mm + offset
        if not hasattr(self, '_filtered_dia'):
            self._filtered_dia = raw_dia
        alpha = 0.3
        self._filtered_dia = alpha * raw_dia + (1 - alpha) * self._filtered_dia
        return round(self._filtered_dia)

    def run(self):
        # first, try opening the port
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
        except SerialException as e:
            self.error_occurred.emit(f"Unable to open {self.port}: {e}")
            self.measurement_complete.emit()
            return

        time.sleep(2)  # give Arduino time to boot

        valid_diams = []
        start = time.time()

        # collect for the configured duration
        while time.time() - start < self.collection_time and self._run_flag:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    try:
                        raw_mm = float(line)
                        if raw_mm > 0:
                            dia = self.calculate_diameter(raw_mm)
                            valid_diams.append(dia)
                            self.diameter_measured.emit(dia)
                    except ValueError:
                        # non‑numeric line; skip
                        pass

                time.sleep(0.005)

            except SerialException as ser_e:
                # "device reports readiness..." error comes through here;
                # just skip this iteration and keep going
                continue

        # emit the median of what we got (if anything)
        if valid_diams:
            self.diameter_measured.emit(float(np.median(valid_diams)))

        # clean up
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
        except Exception:
            pass

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
        # Use 'out' instead of 'fc4' to match saved model
        self.out = nn.Linear(64, 2)  # Changed to match saved model

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)  # Changed to match saved model
        return x
    
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str)
    test_complete_signal = pyqtSignal(np.ndarray, str, str)
    animation_signal = pyqtSignal()
    enable_buttons_signal = pyqtSignal(bool)
    # New signal for real-time classification
    realtime_classification_signal = pyqtSignal(str, str)
    UNKNOWN_THRESHOLD = 0.7

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._testing = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_frame = None
        self.last_classification = ("READY", "")  # (status, recommendation)
        self.classification_timer = QTimer()
        self.classification_timer.timeout.connect(self.classify_current_frame)
        self.classification_timer.start(500)  # Classify every 500ms
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
            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Center-crop to square (360x360) from 480x360
            h, w = frame_gray.shape  # (360, 480)
            side = min(h, w)  # 360
            x0 = (w - side) // 2  # (480-360)//2 = 60
            y0 = (h - side) // 2  # 0
            square = frame_gray[y0:y0+side, x0:x0+side]  # 360x360
            
            # Resize to 128x128
            square_resized = cv2.resize(square, (128, 128))
            
            # HOG features with training parameters
            hog_features = hog(
                square_resized,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                visualize=False,
                feature_vector=True,
                block_norm='L2'
            )
            
            # LMD features
            signal = np.mean(square_resized, axis=0)
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            frequency = np.diff(phase) / (2.0 * np.pi)
            
            # Pad frequency to 128
            if len(frequency) < 128:
                frequency = np.pad(frequency, (0, 128 - len(frequency)))
            elif len(frequency) > 128:
                frequency = frequency[:128]
            
            combined_features = np.concatenate([hog_features, amplitude_envelope, frequency])
            return combined_features
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return np.zeros(2020)

    def classify_current_frame(self):
        if self.model is None or self.last_frame is None:
            return
            
        try:
            features = self.preprocess_image(self.last_frame)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                max_prob = max_prob.item()
            
            # Add unknown detection based on confidence threshold
            if max_prob < self.UNKNOWN_THRESHOLD:
                status = "UNKNOWN"
                recommendation = "Position wheel properly"
            elif predicted.item() == 1:
                status = "FLAW DETECTED"
                recommendation = "For Repair/Replacement"
            else:
                status = "NO FLAW"
                recommendation = "For Constant Monitoring"
            
            # Update last classification
            self.last_classification = (status, recommendation)
            
            # Emit real-time classification
            self.realtime_classification_signal.emit(status, recommendation)

        except Exception as e:
            print(f"Error in real-time classification: {e}")

    def start_test(self):
        self._testing = True
        self.enable_buttons_signal.emit(False)
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
                probs = torch.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                max_prob = max_prob.item()
            
            # Add unknown detection based on confidence threshold
            if max_prob < self.UNKNOWN_THRESHOLD:
                status = "UNKNOWN"
                recommendation = "Position wheel properly"
            elif predicted.item() == 1:
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
        # Higher resolution pipeline (1280x720)
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, "
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
        self.classification_timer.stop()
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
        
        # Add stretch before content to center everything
        self.layout.addStretch(1)
        
        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap('logoV.png')
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(100, Qt.SmoothTransformation))
        self.layout.addWidget(self.logo_label)
        
        # Buttons
        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(20)
        
        # Inspection Button
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
        
        # Calibration Button
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
        
        # Add stretch after content to center everything
        self.layout.addStretch(1)
        self.setLayout(self.layout)

class SelectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 5, 20, 15)  # Reduced top and bottom margins
        self.layout.setSpacing(5)   
        self.layout.addSpacing(15)
        
        # Back Button - made more compact
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: #f0f0f0;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 10px;
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                min-width: 70px;
            }
            QPushButton:hover {
                background: #e60000;
                color: white;
                border-color: #e60000;
            }
            QPushButton:pressed {
                background: #b30000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(0))
        self.layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        
        # Logo - reduced spacing
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(70, Qt.SmoothTransformation))  # Slightly smaller logo
        self.layout.addWidget(self.logo_label)
        
        # Main content container
        content_frame = QFrame()
        content_frame.setStyleSheet("QFrame { background: transparent; }")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)  # Reduced inner margins
        content_layout.setSpacing(10)  # Reduced spacing
        
        # Section title - made more compact
        section_title = QLabel("SELECT INSPECTION DETAILS")
        section_title.setAlignment(Qt.AlignCenter)
        section_title.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 18px;
                color: #333;
                padding-bottom: 3px;
                border-bottom: 2px solid #e60000;
                margin-bottom: 5px;
            }
        """)
        content_layout.addWidget(section_title)
        
        # Train Selection - reduced spacing
        self.train_layout = QVBoxLayout()
        self.train_layout.setSpacing(2)  # Reduced from 0 to 2 for slight separation
        self.train_label = QLabel("Train Number")
        self.train_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 16px;
                color: #555;
                margin-bottom: 2px;
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
                background: #e0e0e0;
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
                color: #e60000;
                margin-top: 2px;
            }
        """)
        self.train_layout.addWidget(self.train_value)
        content_layout.addLayout(self.train_layout)
        
        # Compartment Selection - reduced spacing
        self.compartment_layout = QVBoxLayout()
        self.compartment_layout.setSpacing(2)
        self.compartment_label = QLabel("Compartment Number")
        self.compartment_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 16px;
                color: #555;
                margin-bottom: 2px;
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
        content_layout.addLayout(self.compartment_layout)
        
        # Wheel Selection - reduced spacing
        self.wheel_layout = QVBoxLayout()
        self.wheel_layout.setSpacing(2)
        self.wheel_label = QLabel("Wheel Number")
        self.wheel_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 16px;
                color: #555;
                margin-bottom: 2px;
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
        content_layout.addLayout(self.wheel_layout)
        
        # Start Button - same size but with reduced top margin
        self.start_button = QPushButton("START INSPECTION")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                margin-top: 5px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
        """)
        self.start_button.clicked.connect(self.start_inspection)
        content_layout.addWidget(self.start_button)
        
        content_frame.setLayout(content_layout)
        self.layout.addWidget(content_frame, stretch=1)
        self.setLayout(self.layout)
        
        # Connect signals
        self.train_slider.valueChanged.connect(lambda: self.train_value.setText(str(self.train_slider.value())))
        self.compartment_slider.valueChanged.connect(lambda: self.compartment_value.setText(str(self.compartment_slider.value())))
        self.wheel_slider.valueChanged.connect(lambda: self.wheel_value.setText(str(self.wheel_slider.value())))

    def start_inspection(self):
        self.parent.trainNumber = self.train_slider.value()
        self.parent.compartmentNumber = self.compartment_slider.value()
        self.parent.wheelNumber = self.wheel_slider.value()
        # Update the inspection page's selection label
        self.parent.inspection_page.update_selection_label(
            self.parent.trainNumber, 
            self.parent.compartmentNumber, 
            self.parent.wheelNumber
        )
        self.parent.stacked_widget.setCurrentIndex(2)

class InspectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_diameter = None
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Top bar with back button and selection info
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: #f0f0f0;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 10px;
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                min-width: 70px;
            }
            QPushButton:hover {
                background: #e60000;
                color: white;
                border-color: #e60000;
            }
            QPushButton:pressed {
                background: #b30000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))
        top_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        
        # Selection info
        self.selection_label = QLabel("Train: - | Compartment: - | Wheel: -")
        self.selection_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                color: #555;
                padding: 5px 10px;
                background: #f8f8f8;
                border-radius: 5px;
            }
        """)
        top_layout.addWidget(self.selection_label, alignment=Qt.AlignCenter)
        
        # Spacer to push battery indicator to right
        top_layout.addStretch(1)
        
        # Battery indicator
        self.battery_indicator = BatteryIndicator(compact=True)
        top_layout.addWidget(self.battery_indicator, alignment=Qt.AlignRight)
        
        self.layout.addLayout(top_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left side - Camera feed and status
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        # Camera feed frame
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            QFrame {
                background: #f0f0f0;
                border: 2px solid #ddd;
                border-radius: 10px;
            }
        """)
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(10, 10, 10, 10)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: #000;
                border-radius: 5px;
            }
        """)
        camera_layout.addWidget(self.camera_label)
        
        camera_frame.setLayout(camera_layout)
        left_layout.addWidget(camera_frame)
        
        # Status display
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background: #f8f8f8;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("READY")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 24px;
                color: #333;
                margin-bottom: 5px;
            }
        """)
        status_layout.addWidget(self.status_label)
        
        self.recommendation_label = QLabel("Position wheel for inspection")
        self.recommendation_label.setAlignment(Qt.AlignCenter)
        self.recommendation_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 16px;
                color: #666;
            }
        """)
        status_layout.addWidget(self.recommendation_label)
        
        self.status_frame.setLayout(status_layout)
        left_layout.addWidget(self.status_frame)
        
        content_layout.addLayout(left_layout, stretch=2)
        
        # Right side - Controls and measurements
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # Measurement frame
        measurement_frame = QFrame()
        measurement_frame.setStyleSheet("""
            QFrame {
                background: #f8f8f8;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        measurement_layout = QVBoxLayout()
        
        measurement_title = QLabel("WHEEL MEASUREMENTS")
        measurement_title.setAlignment(Qt.AlignCenter)
        measurement_title.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 18px;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        measurement_layout.addWidget(measurement_title)
        
        # Diameter measurement
        diameter_layout = QVBoxLayout()
        diameter_label = QLabel("Wheel Diameter")
        diameter_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                color: #555;
            }
        """)
        diameter_layout.addWidget(diameter_label)
        
        self.diameter_value = QLabel("-- mm")
        self.diameter_value.setAlignment(Qt.AlignCenter)
        self.diameter_value.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 32px;
                color: #e60000;
                background: white;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
        """)
        diameter_layout.addWidget(self.diameter_value)
        
        self.measure_button = QPushButton("MEASURE DIAMETER")
        self.measure_button.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Montserrat Bold';
                font-size: 14px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.measure_button.clicked.connect(self.measure_diameter)
        diameter_layout.addWidget(self.measure_button)
        
        measurement_layout.addLayout(diameter_layout)
        measurement_frame.setLayout(measurement_layout)
        right_layout.addWidget(measurement_frame)
        
        # Test button
        self.test_button = QPushButton("START TEST")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.test_button.clicked.connect(self.start_test)
        right_layout.addWidget(self.test_button)
        
        # Report button
        self.report_button = QPushButton("GENERATE REPORT")
        self.report_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #111;
            }
            QPushButton:pressed {
                background-color: #000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.report_button.clicked.connect(self.generate_report)
        self.report_button.setEnabled(False)
        right_layout.addWidget(self.report_button)
        
        content_layout.addLayout(right_layout, stretch=1)
        self.layout.addLayout(content_layout, stretch=1)
        self.setLayout(self.layout)

    def update_selection_label(self, train, compartment, wheel):
        self.selection_label.setText(f"Train: {train} | Compartment: {compartment} | Wheel: {wheel}")

    def measure_diameter(self):
        self.measure_button.setEnabled(False)
        self.measure_button.setText("MEASURING...")
        
        # Start diameter measurement thread
        self.parent.serial_thread = SerialReaderThread()
        self.parent.serial_thread.diameter_measured.connect(self.on_diameter_measured)
        self.parent.serial_thread.measurement_complete.connect(self.on_measurement_complete)
        self.parent.serial_thread.error_occurred.connect(self.on_measurement_error)
        self.parent.serial_thread.start()

    def on_diameter_measured(self, diameter):
        self.current_diameter = diameter
        self.diameter_value.setText(f"{diameter:.1f} mm")

    def on_measurement_complete(self):
        self.measure_button.setEnabled(True)
        self.measure_button.setText("MEASURE DIAMETER")
        # Enable report button only if we have a measurement
        if self.current_diameter is not None:
            self.report_button.setEnabled(True)

    def on_measurement_error(self, error_msg):
        print(f"Measurement error: {error_msg}")
        self.diameter_value.setText("ERROR")
        self.measure_button.setEnabled(True)
        self.measure_button.setText("MEASURE DIAMETER")

    def start_test(self):
        self.test_button.setEnabled(False)
        self.parent.camera_thread.start_test()

    def generate_report(self):
        if self.current_diameter is None:
            QMessageBox.warning(self, "Warning", "Please measure diameter first")
            return
            
        # Get the current status and recommendation
        status, recommendation = self.parent.camera_thread.last_classification
        
        # Convert current frame to base64
        if self.parent.camera_thread.last_frame is not None:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', self.parent.camera_thread.last_frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send report to backend
            success = send_report_to_backend(
                status=status,
                recommendation=recommendation,
                image_base64=image_base64,
                name="Operator",  # You can customize this
                trainNumber=self.parent.trainNumber,
                compartmentNumber=self.parent.compartmentNumber,
                wheelNumber=self.parent.wheelNumber,
                wheel_diameter=self.current_diameter,
                token=None  # Add authentication token if needed
            )
            
            if success:
                QMessageBox.information(self, "Success", "Report generated and sent successfully!")
            else:
                QMessageBox.warning(self, "Warning", "Failed to send report. Please try again.")
        else:
            QMessageBox.warning(self, "Warning", "No image captured. Please run test first.")

class CalibrationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: #f0f0f0;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 10px;
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                min-width: 70px;
            }
            QPushButton:hover {
                background: #e60000;
                color: white;
                border-color: #e60000;
            }
            QPushButton:pressed {
                background: #b30000;
            }
        """)
        self.back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(0))
        self.layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        
        # Title
        title = QLabel("CALIBRATION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 24px;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        self.layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Place the 625mm calibration ring on the measurement pads\n"
            "2. Click 'Calibrate 625mm' and wait for measurement\n"
            "3. Replace with the 700mm calibration ring\n"
            "4. Click 'Calibrate 700mm' and wait for measurement\n"
            "5. Click 'Save Calibration' to store the values"
        )
        instructions.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 14px;
                color: #555;
                background: #f8f8f8;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
        """)
        instructions.setWordWrap(True)
        self.layout.addWidget(instructions)
        
        # Calibration values display
        values_frame = QFrame()
        values_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        values_layout = QVBoxLayout()
        
        values_title = QLabel("Current Calibration Values")
        values_title.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 16px;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        values_layout.addWidget(values_title)
        
        self.values_display = QLabel("625mm: -- mm\n700mm: -- mm")
        self.values_display.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat';
                font-size: 14px;
                color: #666;
                background: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        values_layout.addWidget(self.values_display)
        
        values_frame.setLayout(values_layout)
        self.layout.addWidget(values_frame)
        
        # Calibration buttons
        buttons_layout = QHBoxLayout()
        
        self.calibrate_625_btn = QPushButton("Calibrate 625mm")
        self.calibrate_625_btn.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Montserrat Bold';
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.calibrate_625_btn.clicked.connect(lambda: self.start_calibration(625))
        buttons_layout.addWidget(self.calibrate_625_btn)
        
        self.calibrate_700_btn = QPushButton("Calibrate 700mm")
        self.calibrate_700_btn.setStyleSheet("""
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Montserrat Bold';
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.calibrate_700_btn.clicked.connect(lambda: self.start_calibration(700))
        buttons_layout.addWidget(self.calibrate_700_btn)
        
        self.layout.addLayout(buttons_layout)
        
        # Save button
        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #111;
            }
            QPushButton:pressed {
                background-color: #000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        self.layout.addWidget(self.save_btn)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        
        # Initialize calibration values
        self.calibration_values = {625: None, 700: None}
        self.load_current_values()

    def load_current_values(self):
        try:
            if os.path.exists("calibration_values.txt"):
                with open("calibration_values.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(':')
                        if len(parts) < 2:
                            continue
                        key = parts[0].strip()
                        value = parts[1].strip()
                        try:
                            if key == "625mm":
                                self.calibration_values[625] = float(value)
                            elif key == "700mm":
                                self.calibration_values[700] = float(value)
                        except ValueError:
                            continue
                self.update_values_display()
        except Exception as e:
            print(f"Error loading calibration values: {e}")

    def update_values_display(self):
        text = f"625mm: {self.calibration_values[625] or '--'} mm\n700mm: {self.calibration_values[700] or '--'} mm"
        self.values_display.setText(text)
        
        # Enable save button only if both values are set
        if self.calibration_values[625] is not None and self.calibration_values[700] is not None:
            self.save_btn.setEnabled(True)

    def start_calibration(self, size):
        # Disable buttons during measurement
        self.calibrate_625_btn.setEnabled(False)
        self.calibrate_700_btn.setEnabled(False)
        
        # Start calibration measurement
        self.calibration_thread = CalibrationSerialThread()
        self.calibration_thread.distance_measured.connect(lambda dist: self.on_distance_measured(size, dist))
        self.calibration_thread.measurement_complete.connect(lambda: self.on_calibration_complete(size))
        self.calibration_thread.error_occurred.connect(self.on_calibration_error)
        self.calibration_thread.start()

    def on_distance_measured(self, size, distance):
        # Store the measured distance for this calibration size
        self.calibration_values[size] = distance
        self.update_values_display()

    def on_calibration_complete(self, size):
        # Re-enable buttons
        self.calibrate_625_btn.setEnabled(True)
        self.calibrate_700_btn.setEnabled(True)
        
        QMessageBox.information(self, "Success", f"Calibration for {size}mm completed!")

    def on_calibration_error(self, error_msg):
        # Re-enable buttons
        self.calibrate_625_btn.setEnabled(True)
        self.calibrate_700_btn.setEnabled(True)
        
        QMessageBox.warning(self, "Error", f"Calibration failed: {error_msg}")

    def save_calibration(self):
        try:
            with open("calibration_values.txt", "w") as f:
                f.write(f"625mm: {self.calibration_values[625]}\n")
                f.write(f"700mm: {self.calibration_values[700]}\n")
                
                # Calculate and save slope and offset
                slope = (700.0 - 625.0) / (self.calibration_values[700] - self.calibration_values[625])
                offset = 700.0 - slope * self.calibration_values[700]
                f.write(f"M_SLOPE: {slope}\n")
                f.write(f"B_OFFS: {offset}\n")
            
            QMessageBox.information(self, "Success", "Calibration values saved successfully!")
            
            # Reload calibration in serial thread
            if hasattr(self.parent, 'serial_thread') and self.parent.serial_thread:
                self.parent.serial_thread.load_calibration_values()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save calibration: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.trainNumber = None
        self.compartmentNumber = None
        self.wheelNumber = None
        self.camera_thread = None
        self.serial_thread = None
        self.battery_thread = None
        self.setup_ui()
        self.setup_camera()
        self.setup_battery_monitor()
        
    def setup_ui(self):
        self.setWindowTitle("Wheel Inspection System")
        self.setMinimumSize(1024, 768)
        
        # Set window icon
        if os.path.exists('logoV.png'):
            self.setWindowIcon(QIcon('logoV.png'))
        
        # Load fonts
        self.load_fonts()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.home_page = HomePage(self)
        self.selection_page = SelectionPage(self)
        self.inspection_page = InspectionPage(self)
        self.calibration_page = CalibrationPage(self)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.selection_page)
        self.stacked_widget.addWidget(self.inspection_page)
        self.stacked_widget.addWidget(self.calibration_page)
        
        # Set initial page
        self.stacked_widget.setCurrentIndex(0)

    def load_fonts(self):
        font_paths = [
            'Montserrat-Bold.ttf',
            'Montserrat-SemiBold.ttf',
            'Montserrat-Regular.ttf'
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_id = QFontDatabase.addApplicationFont(font_path)
                if font_id == -1:
                    print(f"Failed to load font: {font_path}")

    def setup_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_camera_feed)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.test_complete_signal.connect(self.on_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_status_animation)
        self.camera_thread.enable_buttons_signal.connect(self.enable_test_button)
        self.camera_thread.realtime_classification_signal.connect(self.update_realtime_status)
        self.camera_thread.start()

    def setup_battery_monitor(self):
        self.battery_thread = BatteryMonitorThread()
        self.battery_thread.battery_updated.connect(self.update_battery_display)
        self.battery_thread.start()

    def update_battery_display(self, voltage, percentage):
        # Update battery indicator on inspection page
        if hasattr(self, 'inspection_page'):
            self.inspection_page.battery_indicator.update_battery(voltage, percentage)

    def update_camera_feed(self, image):
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.inspection_page.camera_label.width(),
            self.inspection_page.camera_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.inspection_page.camera_label.setPixmap(scaled_pixmap)

    def update_status(self, status, recommendation):
        self.inspection_page.status_label.setText(status)
        self.inspection_page.recommendation_label.setText(recommendation)
        
        # Update status frame color based on status
        if status == "FLAW DETECTED":
            color = "#ff4444"
        elif status == "NO FLAW":
            color = "#44ff44"
        elif status == "UNKNOWN":
            color = "#ffaa44"
        else:
            color = "#f8f8f8"
            
        self.inspection_page.status_frame.setStyleSheet(f"""
            QFrame {{
                background: {color};
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
            }}
        """)

    def update_realtime_status(self, status, recommendation):
        # Only update if not in the middle of a test
        if not self.camera_thread._testing:
            self.update_status(status, recommendation)

    def on_test_complete(self, frame, status, recommendation):
        # This is called when a test is explicitly started
        self.update_status(status, recommendation)

    def trigger_status_animation(self):
        # Simple animation by temporarily changing background
        original_style = self.inspection_page.status_frame.styleSheet()
        self.inspection_page.status_frame.setStyleSheet("""
            QFrame {
                background: #ffff44;
                border: 2px solid #ffaa00;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        QTimer.singleShot(300, lambda: self.inspection_page.status_frame.setStyleSheet(original_style))

    def enable_test_button(self, enabled):
        self.inspection_page.test_button.setEnabled(enabled)

    def closeEvent(self, event):
        # Clean up threads
        if self.camera_thread:
            self.camera_thread.stop()
        if self.serial_thread:
            self.serial_thread.stop()
        if self.battery_thread:
            self.battery_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide styles
    app.setStyleSheet("""
        QMainWindow {
            background: #f5f5f5;
        }
        QWidget {
            font-family: 'Montserrat';
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())