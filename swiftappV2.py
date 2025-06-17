import sys
import os
import time
import cv2
import base64
import torch
import numpy as np
import requests
import smbus
from ina219 import INA219
from ina219 import DeviceRangeError
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QFrame, QMessageBox, QSizePolicy,
                            QGridLayout, QStackedWidget, QSlider, QStyle)
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
    LIFT_OFF_MM = 39.0  # sensor→lever gap when off-wheel
    
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
                        if "700mm:" in line:
                            self.CAL_700_RAW = float(line.split(":")[1].strip())
                        elif "625mm:" in line:
                            self.CAL_625_RAW = float(line.split(":")[1].strip())
                        elif "M_SLOPE:" in line:
                            self.M_SLOPE = float(line.split(":")[1].strip())
                        elif "B_OFFS:" in line:
                            self.B_OFFS = float(line.split(":")[1].strip())
                            
                # Recalculate in case file was incomplete
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
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # allow Arduino to wake up

            valid_diameters = []
            start_time = time.time()

            # ←── COLLECT for self.collection_time seconds
            while time.time() - start_time < self.collection_time and self._run_flag:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    try:
                        raw_mm = float(line)                     # treat incoming as raw mm
                        if raw_mm > 0:
                            dia = self.calculate_diameter(raw_mm) # apply your formula
                            valid_diameters.append(dia)
                            self.diameter_measured.emit(dia)
                    except ValueError:
                        pass
                time.sleep(0.005)  # tighter polling, so we catch every 200 ms frame

            # Emit the median of all collected diameters
            if valid_diameters:
                median_d = float(np.median(valid_diameters))
                self.diameter_measured.emit(median_d)

        except Exception as e:
            self.error_occurred.emit(str(e))

        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
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
    # New signal for real-time classification
    realtime_classification_signal = pyqtSignal(str, str)

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
                _, predicted = torch.max(outputs, 1)
            
            if predicted.item() == 1:
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
            # Use the last classification result
            status, recommendation = self.last_classification
            
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
        self.parent.inspection_page.update_selection_label()
        self.parent.stacked_widget.setCurrentIndex(2)

class InspectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.captured_image = None  # Add this to store captured image
        self.is_captured_mode = False  # Flag for captured image display
        self.setup_ui()
        self.setup_animations()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        self.layout.addSpacing(35)

        # Camera Panel - Top section
        self.camera_panel = QFrame()
        self.camera_panel.setStyleSheet("QFrame { background: white; border: 5px solid transparent; }")
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: 5px solid transparent;
            }
        """)
        self.camera_layout.addWidget(self.camera_label)
        
        # Add real-time status indicator HERE after camera_label
        self.realtime_status_indicator = QLabel("READY")
        self.realtime_status_indicator.setAlignment(Qt.AlignCenter)
        self.realtime_status_indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat Regular';
                font-size: 14px;
                padding-top: 5px;
            }
        """)
        self.realtime_status_indicator.setVisible(True)
        self.camera_layout.addWidget(self.realtime_status_indicator, alignment=Qt.AlignBottom | Qt.AlignCenter)

        self.camera_panel.setLayout(self.camera_layout)
        self.layout.addWidget(self.camera_panel, stretch=1)  # Camera takes more space
        
        # Control Panel - Bottom section
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
                font-family: 'Montserrat Regular';
                font-size: 14px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
        """)

        self.diameter_label = QLabel("Wheel Diameter: -")
        self.diameter_label.setAlignment(Qt.AlignCenter)
        self.diameter_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
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
        
        # Button Panel - Horizontal layout for buttons
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 10, 0, 10)
        self.button_layout.setSpacing(10)
        
        # Button style
        button_style = """
            QPushButton {
                background-color: %s;
                color: white;
                border: none;
                padding: 15px 25px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                border-radius: 8px;
                min-width: 140px;
                min-height: 80px;
                max-width: 140px;
            }
            QPushButton:hover { background-color: %s; }
            QPushButton:pressed { background-color: %s; }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """
        
        self.detect_btn = QPushButton("CAPTURE\nFLAWS")
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setStyleSheet(button_style % ("#e60000", "#cc0000", "#b30000"))
        
        self.measure_btn = QPushButton("MEASURE\nDIAMETER")
        self.measure_btn.setEnabled(False)
        self.measure_btn.setCursor(Qt.PointingHandCursor)
        self.measure_btn.setStyleSheet(button_style % ("#0066cc", "#0055aa", "#004488"))
        
        self.save_btn = QPushButton("SAVE\nREPORT")
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet(button_style % ("#FFC107", "#FFB300", "#FFA000"))
        
        self.reset_btn = QPushButton("NEW\nINSPECTION")
        self.reset_btn.setVisible(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(button_style % ("#000000", "#333333", "#222222"))
        
        # Center the buttons horizontally
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.detect_btn)
        self.button_layout.addWidget(self.measure_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_layout.addStretch(1)
        
        self.button_panel.setLayout(self.button_layout)
        self.control_layout.addWidget(self.button_panel)
        
        self.control_panel.setLayout(self.control_layout)
        self.layout.addWidget(self.control_panel, stretch=0)  # Control panel takes less space
        
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
        self.calibration_values = {"700mm": None, "625mm": None}
        self.calibration_timestamps = {"700mm": None, "625mm": None}  # New dictionary for timestamps
        self.current_reading = None
        self.load_calibration_values()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(30, 20, 30, 30)
        self.layout.setSpacing(15)
        
        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background: #f0f0f0;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px 15px;
                font-family: 'Montserrat SemiBold';
                font-size: 14px;
                min-width: 80px;
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
        
        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap.scaledToHeight(80, Qt.SmoothTransformation))
        self.layout.addWidget(self.logo_label)
        
        # Main Title
        self.title_label = QLabel("Calibration")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 28px;
                color: #333;
                padding: 5px 0;
            }
        """)
        self.layout.addWidget(self.title_label)
        
        # 700mm Calibration Section
        self.calib_700_group = self.create_calibration_group("700 mm Train Wheel", "1st Calibration")
        self.layout.addWidget(self.calib_700_group)
        
        # 625mm Calibration Section
        self.calib_625_group = self.create_calibration_group("625 mm Train Wheel", "2nd Calibration")
        self.layout.addWidget(self.calib_625_group)
        
        # Status Label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Regular';
                font-size: 14px;
                color: #666;
                padding: 10px 0;
            }
        """)
        self.layout.addWidget(self.status_label)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)

    def create_calibration_group(self, title, button_text):
        group = QFrame()
        group.setStyleSheet("QFrame { background: #f8f8f8; border-radius: 10px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Bold';
                font-size: 18px;
                color: #333;
            }
        """)
        layout.addWidget(title_label, alignment=Qt.AlignCenter)
        
        # Sensor Reading
        reading_label = QLabel("Distance: -")
        reading_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat SemiBold';
                font-size: 16px;
                color: #555;
            }
        """)
        layout.addWidget(reading_label, alignment=Qt.AlignCenter)
        
        # Last Calibration Timestamp (new)
        timestamp_label = QLabel("Last calibrated: Never")
        timestamp_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat Regular';
                font-size: 12px;
                color: #777;
                font-style: italic;
            }
        """)
        layout.addWidget(timestamp_label, alignment=Qt.AlignCenter)
        
        # Calibration Button
        calib_button = QPushButton(button_text)
        calib_button.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Montserrat Bold';
                font-size: 16px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #0055aa;
            }
            QPushButton:pressed {
                background-color: #004488;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """)
        
        # Store references to update later
        if "700" in title:
            self.calib_700_reading = reading_label
            self.calib_700_button = calib_button
            self.calib_700_timestamp = timestamp_label  # New reference
            calib_button.clicked.connect(lambda: self.start_measurement("700mm"))
        else:
            self.calib_625_reading = reading_label
            self.calib_625_button = calib_button
            self.calib_625_timestamp = timestamp_label  # New reference
            calib_button.clicked.connect(lambda: self.start_measurement("625mm"))
            
        layout.addWidget(calib_button)
        
        group.setLayout(layout)
        return group

    def start_measurement(self, wheel_type):
        # Disable both buttons during measurement
        self.calib_700_button.setEnabled(False)
        self.calib_625_button.setEnabled(False)
        
        # Clear previous readings
        if wheel_type == "700mm":
            self.calib_700_reading.setText("Measuring...")
        else:
            self.calib_625_reading.setText("Measuring...")
        
        try:
            self.serial_thread = CalibrationSerialThread()
            self.serial_thread.distance_measured.connect(lambda dist: self.update_reading(wheel_type, dist))
            self.serial_thread.measurement_complete.connect(lambda: self.on_measurement_complete(wheel_type))
            self.serial_thread.error_occurred.connect(self.handle_serial_error)
            self.serial_thread.start()

        except Exception as e:
            print(f"Serial connection error: {e}")
            self.handle_serial_error(f"Serial error: {str(e)}")

    def update_reading(self, wheel_type, distance):
        self.current_reading = distance
        if wheel_type == "700mm":
            self.calib_700_reading.setText(f"Distance: {distance} mm")
        else:
            self.calib_625_reading.setText(f"Distance: {distance} mm")

    def on_measurement_complete(self, wheel_type):
        if self.current_reading is not None:
            self.calibration_values[wheel_type] = self.current_reading
            # Update timestamp with current date and time in military format
            current_time = time.strftime("%Y-%m-%d %H:%M")  # Changed to 24-hour format without seconds
            self.calibration_timestamps[wheel_type] = current_time
            
            # Update the timestamp label
            if wheel_type == "700mm":
                self.calib_700_timestamp.setText(f"Last calibrated: {current_time}")
            else:
                self.calib_625_timestamp.setText(f"Last calibrated: {current_time}")
                
            self.status_label.setText(f"{wheel_type} calibrated at {self.current_reading} mm")
            
            # Update the calibration constants in SerialReaderThread
            if wheel_type == "700mm":
                SerialReaderThread.CAL_700_RAW = self.current_reading
            else:
                SerialReaderThread.CAL_625_RAW = self.current_reading
                
            # Recalculate the slope and offset
            SerialReaderThread.M_SLOPE = (700.0 - 625.0) / (
                SerialReaderThread.CAL_700_RAW - SerialReaderThread.CAL_625_RAW
            )
            SerialReaderThread.B_OFFS = 700.0 - SerialReaderThread.M_SLOPE * SerialReaderThread.CAL_700_RAW
            
            self.save_calibration_values()
        
        # Re-enable buttons
        self.calib_700_button.setEnabled(True)
        self.calib_625_button.setEnabled(True)

    def handle_serial_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        # Re-enable buttons on error
        self.calib_700_button.setEnabled(True)
        self.calib_625_button.setEnabled(True)

    def save_calibration_values(self):
        # Save to file with the new format that includes recalculated constants and timestamps
        print("Calibration values:", self.calibration_values)
        with open("calibration_values.txt", "w") as f:
            f.write(f"700mm: {self.calibration_values['700mm']}\n")
            f.write(f"625mm: {self.calibration_values['625mm']}\n")
            f.write(f"M_SLOPE: {SerialReaderThread.M_SLOPE}\n")
            f.write(f"B_OFFS: {SerialReaderThread.B_OFFS}\n")
            # Save timestamps if they exist
            if self.calibration_timestamps['700mm']:
                f.write(f"700mm_timestamp: {self.calibration_timestamps['700mm']}\n")
            if self.calibration_timestamps['625mm']:
                f.write(f"625mm_timestamp: {self.calibration_timestamps['625mm']}\n")

    def load_calibration_values(self):
        try:
            if os.path.exists("calibration_values.txt"):
                with open("calibration_values.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "700mm:" in line and not "timestamp" in line:
                            self.calibration_values['700mm'] = float(line.split(":")[1].strip())
                            self.calib_700_reading.setText(f"Distance: {self.calibration_values['700mm']} mm")
                        elif "625mm:" in line and not "timestamp" in line:
                            self.calibration_values['625mm'] = float(line.split(":")[1].strip())
                            self.calib_625_reading.setText(f"Distance: {self.calibration_values['625mm']} mm")
                        elif "700mm_timestamp:" in line:
                            self.calibration_timestamps['700mm'] = line.split(":")[1].strip()
                            self.calib_700_timestamp.setText(f"Last calibrated: {self.calibration_timestamps['700mm']}")
                        elif "625mm_timestamp:" in line:
                            self.calibration_timestamps['625mm'] = line.split(":")[1].strip()
                            self.calib_625_timestamp.setText(f"Last calibrated: {self.calibration_timestamps['625mm']}")
                        elif "M_SLOPE:" in line:
                            SerialReaderThread.M_SLOPE = float(line.split(":")[1].strip())
                        elif "B_OFFS:" in line:
                            SerialReaderThread.B_OFFS = float(line.split(":")[1].strip())
        except Exception as e:
            print(f"Error loading calibration values: {e}")

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel Inspection")
        self.setWindowIcon(QIcon("logo.png"))
        
        # Initialize attributes first
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.current_distance = 0
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None
        self.captured_image = None  # Add this to store captured image
        
        # Initialize UI components to avoid attribute errors
        self.battery_indicator = None
        self.stacked_widget = None
        
        # Set up UI
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
        
        # Create battery monitor and indicator
        self.battery_monitor = BatteryMonitorThread()
        self.battery_indicator = BatteryIndicator()
        
        # Add battery indicator to top right corner
        self.battery_indicator.setParent(self.central_widget)
        
        # Now show the window
        self.setMinimumSize(480, 800)  # Set a reasonable minimum size
        QApplication.processEvents()  # Allow initial layout calculations
        self.showNormal()  # Show normal first
        QApplication.processEvents()  # Process any pending events
        self.showFullScreen()  # Then go fullscreen
        
        # Position battery indicator after window is shown
        self.battery_indicator.move(self.width() - 100, 10)
        
        # Connect signals
        self.battery_monitor.battery_updated.connect(self.battery_indicator.update_battery)
        self.battery_monitor.start()
        self.inspection_page.reset_btn.clicked.connect(self.reset_ui)

    def resizeEvent(self, event):
        # Ensure the layout stays stable during resizing
        if self.battery_indicator:
            self.battery_indicator.move(self.width() - 100, 10)
        if self.stacked_widget:
            self.stacked_widget.updateGeometry()
            self.stacked_widget.adjustSize()
        super().resizeEvent(event)

    def showEvent(self, event):
        # Ensure proper layout when showing
        if self.stacked_widget:
            self.stacked_widget.updateGeometry()
            self.stacked_widget.adjustSize()
        super().showEvent(event)

    def setup_camera_thread(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.test_complete_signal.connect(self.handle_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_animation)
        self.camera_thread.enable_buttons_signal.connect(self.set_buttons_enabled)
        # Connect real-time classification signal
        self.camera_thread.realtime_classification_signal.connect(self.update_realtime_status)
        self.camera_thread.start()

    def update_realtime_status(self, status, recommendation):
        """Update the real-time classification status in the UI"""
        self.inspection_page.realtime_status_indicator.setText(status)
        
        # Update status color based on classification
        if status == "FLAW DETECTED":
            self.inspection_page.realtime_status_indicator.setStyleSheet("""
                QLabel {
                    color: red;
                    font-family: 'Montserrat SemiBold';
                    font-size: 14px;
                    background-color: rgba(0,0,0,0.5);
                    padding: 2px 5px;
                    border-radius: 5px;
                }
            """)
        elif status == "NO FLAW":
            self.inspection_page.realtime_status_indicator.setStyleSheet("""
                QLabel {
                    color: #00CC00;
                    font-family: 'Montserrat SemiBold';
                    font-size: 14px;
                    background-color: rgba(0,0,0,0.5);
                    padding: 2px 5px;
                    border-radius: 5px;
                }
            """)
        else:
            self.inspection_page.realtime_status_indicator.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-family: 'Montserrat Regular';
                    font-size: 14px;
                }
            """)

    def update_image(self, qt_image):
        if self.captured_image:
            # Display captured image if available
            self.inspection_page.camera_label.setPixmap(
                QPixmap.fromImage(self.captured_image).scaled(
                self.inspection_page.camera_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        else:
            # Otherwise show live feed
            self.inspection_page.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.inspection_page.camera_label.size(), 
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
                    border: 5px solid red;
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
                    border: 5px solid #00CC00;
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
                    border: 5px solid transparent;
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
                border: 5px solid transparent;
            }
        """)
        self.inspection_page.diameter_label.hide()
        
        # Immediately disable the button for visual feedback
        self.inspection_page.detect_btn.setEnabled(False)
        self.inspection_page.measure_btn.setEnabled(False)
        self.inspection_page.save_btn.setEnabled(False)
        
        self.camera_thread.start_test()

        if self.camera_thread.last_frame is not None:
            frame = self.camera_thread.last_frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            self.captured_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Hide the real-time status indicator after capturing
        self.inspection_page.realtime_status_indicator.hide()

    def update_diameter(self, diameter):
        """Update the UI with the measured diameter"""
        self.current_distance = diameter
        diameter_text = f"Wheel Diameter: {diameter:.1f} mm"
        self.inspection_page.diameter_label.setText(diameter_text)
        
        # Set color and font based on diameter value
        if diameter <= 620:
            color = "#FF0000"  # Red
        else:
            color = "#00CC00"# Green
            
        self.inspection_page.diameter_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-family: 'Montserrat Bold';
                font-size: 14px;
            }}
        """)
        
        # Enable save button if we have both test result and measurement
        if hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"]:
            self.inspection_page.save_btn.setEnabled(True)
            
    def measure_diameter(self):
        self.inspection_page.diameter_label.setText("Measuring...")
        self.inspection_page.diameter_label.show()

        # Disable all buttons during measurement
        self.inspection_page.detect_btn.setEnabled(False)
        self.inspection_page.measure_btn.setEnabled(False)
        self.inspection_page.save_btn.setEnabled(False)

        try:
            self.serial_thread = SerialReaderThread()
            self.serial_thread.diameter_measured.connect(self.update_diameter)  # Connect to the new method
            self.serial_thread.measurement_complete.connect(self.on_diameter_measurement_complete)
            self.serial_thread.error_occurred.connect(self.handle_measurement_error)
            self.serial_thread.start()

        except Exception as e:
            print(f"Serial connection error: {e}")
            self.handle_measurement_error(f"Serial error: {str(e)}")

    def handle_measurement_error(self, error_msg):  # New method to handle errors
        print(f"Measurement error: {error_msg}")
        self.inspection_page.diameter_label.setText("Measurement Error")
        self.on_diameter_measurement_complete()

    def on_diameter_measurement_complete(self):  # Renamed from on_measurement_complete
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
        self.captured_image = None  # Clear captured image flag

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
        msg = QMessageBox(self)
        msg.setWindowTitle("Save Report")
        msg.setText("Save this inspection report?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border: 1px solid #ddd;
                font-family: 'Montserrat Regular';
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
        msg.setWindowModality(Qt.ApplicationModal)
        
        # Center the message box on screen
        msg.setWindowModality(Qt.WindowModal)
        msg.setGeometry(
            self.geometry().center().x() - 150,
            self.geometry().center().y() - 75,
            300,
            150
        )
        
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
        self.inspection_page.update_selection_label()
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
                font-family: 'Montserrat Regular';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        self.inspection_page.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: 5px solid transparent;
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
        self.current_distance = 0
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None
        self.captured_image = None

        self.inspection_page.realtime_status_indicator.show()
        
        # Reload the model for next use
        self.camera_thread.load_model()

    def closeEvent(self, event):
        self.camera_thread.stop()
        if hasattr(self, 'sensor_thread'):
            self.sensor_thread.stop()
            self.battery_monitor.stop()
        event.accept()

if __name__ == "__main__":
    os.environ["QT_QUICK_BACKEND"] = "software"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load Montserrat font if available
    font_db = QFontDatabase()
    if "Montserrat Regular" not in font_db.families():
        # Try to load the font from file if not found
        font_paths = {
            "Montserrat Regular": "Montserrat-Regular.ttf",
            "Montserrat Bold": "Montserrat-Bold.ttf",
            "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
            "Montserrat Black": "Montserrat-Black.ttf"
        }
        for font_name, path in font_paths.items():
            if not os.path.exists(path):
                print(f"Font file not found: {path}")
                continue
            font_id = font_db.addApplicationFont(path)
            if font_id == -1:
                print(f"Failed to load font: {path}")
    
    # Set application font
    font = QFont("Montserrat Regular", 12)
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
