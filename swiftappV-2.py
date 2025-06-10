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

# VL53L0X library is not directly used in the Jetson code, as the Arduino handles the sensor communication.
# The Python script communicates with the Arduino via serial to get distance measurements.

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

    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        super().__init__()
        self._run_flag = True
        self.port = port
        self.baudrate = baudrate

    def run(self):
        try:
            # Attempt to open serial port
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                print(f"Connected to {self.port} at {self.baudrate} baud")
                
                # Clear any existing data in buffers
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # Allow time for Arduino to initialize and send its initial messages
                time.sleep(2)
                
                # Read and discard any initial messages from Arduino (e.g., "Initializing VL53L0X sensor...")
                while ser.in_waiting:
                    ser.readline()

                # Send request for reading to Arduino
                ser.write(b'R\n') # Send 'R' followed by a newline as Arduino expects Serial.read()
                
                # Read with timeout
                start_time = time.time()
                while time.time() - start_time < 5:  # Increased timeout to 5 seconds for robustness
                    if ser.in_waiting:
                        line = ser.readline().decode('ascii', errors='ignore').strip()
                        if line:
                            print(f"Received from Arduino: {line}")
                            try:
                                distance = int(line)
                                if distance > 0:  # Valid distance measurement
                                    self.distance_measured.emit(distance)
                                    self.measurement_complete.emit()
                                    return
                                elif distance == 0: # No new measurement available from Arduino
                                    self.error_occurred.emit("No new distance measurement available from Arduino.")
                                    break
                                elif distance == -1:
                                    self.error_occurred.emit("Arduino: Sensor initialization error.")
                                    break
                                elif distance == -2:
                                    self.error_occurred.emit("Arduino: Sensor timeout occurred.")
                                    break
                            except ValueError:
                                # This handles cases where Arduino sends non-integer debug messages
                                print(f"Non-integer data received from Arduino: {line}")
                                continue  # Skip non-integer lines and continue waiting for a valid measurement
                
                self.error_occurred.emit("No valid distance measurement received from Arduino within timeout.")
                
        except serial.SerialException as e:
            self.error_occurred.emit(f"Serial port error: {str(e)}. Make sure Arduino is connected and the correct port is selected.")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error in DistanceSensorThread: {str(e)}")
        finally:
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

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel Inspection")
        self.setWindowIcon(QIcon("logo.png"))
        self.showFullScreen()
            
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.current_distance = 680 # Initial dummy value
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
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Camera Panel
        self.camera_panel = QFrame()
        self.camera_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet("QLabel { background: black; border: none; }")
        
        self.camera_layout.addWidget(self.camera_label)
        self.camera_panel.setFixedHeight(360)
        self.camera_panel.setFixedWidth(480)
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_layout.setAlignment(Qt.AlignCenter)
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
        self.status_layout.setContentsMargins(5, 5, 5, 5)
        
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
        self.measure_btn.setEnabled(True) # Enabled by default for testing serial communication
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
        self.control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.control_panel.setLayout(self.control_layout)
        
        self.content_layout.addWidget(self.camera_panel)
        self.content_layout.addWidget(self.control_panel)
        
        self.main_layout.addLayout(self.content_layout)
        
        self.setup_animations()
        self.setup_camera_thread()
        self.connect_signals()
        self.setup_number_controls()

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
        # Assuming a simple linear relationship for demonstration. Adjust as needed.
        # For example, if 680mm distance corresponds to 920mm diameter, and 700mm distance to 900mm diameter
        # This is a placeholder and needs actual calibration data.
        # Let's assume a simple inverse relationship for now: smaller distance means larger diameter
        # And a baseline: if distance is 680mm, diameter is 920mm
        # If distance increases by 1mm, diameter decreases by X mm
        # Example: if distance increases by 1mm, diameter decreases by 2mm
        # diameter = 920 - (distance - 680) * 2
        
        # A more robust approach would be to use a lookup table or a more complex regression model
        # For now, let's just display the distance directly as diameter for simplicity, or a fixed value.
        # self.diameter_label.setText(f"Wheel Diameter: {self.current_distance} mm")
        
        # Placeholder for actual diameter calculation based on distance
        # This needs to be calibrated with actual wheel diameters and sensor distances
        # For now, let's use a dummy calculation or just display the distance
        
        # Example: If the sensor is 680mm away from a 920mm wheel, and we want to show 920mm
        # If the sensor reads 680, then diameter is 920
        # If the sensor reads 670, then diameter is 930 (closer, larger wheel)
        # If the sensor reads 690, then diameter is 910 (further, smaller wheel)
        
        # Let's assume a simple linear mapping for now: 1mm change in distance = 1mm change in diameter (inverse)
        # Base distance = 680mm, Base diameter = 920mm
        # diameter = 920 + (680 - self.current_distance)
        
        # For a more realistic scenario, you'd have a calibration curve or formula.
        # For this example, let's just show a fixed diameter if the distance is within a certain range,
        # or directly show the distance as diameter for now.
        
        # Let's assume a target distance of 680mm for a standard wheel of 920mm diameter.
        # If the measured distance is `d`, and the reference distance is `d_ref` (680mm) for a reference diameter `dia_ref` (920mm).
        # A simple inverse proportionality: `diameter = dia_ref * (d_ref / d)`
        # This might not be perfectly linear, but it's a starting point.
        
        if self.current_distance > 0:
            # Example: if 680mm distance corresponds to 920mm diameter
            # Let's use a simple linear interpolation for demonstration purposes.
            # You would replace this with your actual calibration data.
            
            # Define two calibration points (distance, diameter)
            # Point 1: (d1, dia1) - e.g., (650mm, 950mm)
            # Point 2: (d2, dia2) - e.g., (700mm, 880mm)
            
            # For simplicity, let's just use a direct display of distance for now,
            # or a dummy calculation that shows some variation.
            
            # Dummy calculation: if distance is 680, diameter is 920. Each 10mm deviation changes diameter by 5mm.
            # If distance is less than 680, diameter is larger. If more, diameter is smaller.
            
            diameter_calculated = 920 + (680 - self.current_distance) * 0.5 # Adjust multiplier as needed
            self.diameter_label.setText(f"Wheel Diameter: {diameter_calculated:.2f} mm")
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

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
        self.animation.setDuration(1000) # 1 second
        
        # Start position (current position)
        current_rect = self.analyzing_label.geometry()
        self.animation.setStartValue(current_rect)
        
        # End position (move slightly to the right and back)
        end_rect = current_rect.translated(10, 0) # Move 10 pixels right
        self.animation.setEndValue(end_rect)
        
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.setLoopCount(2) # Play twice (right and back)
        self.animation.finished.connect(self.hide_analyzing_label)
        self.animation.start()

    def hide_analyzing_label(self):
        self.analyzing_label.hide()

    def enable_buttons(self, enable):
        self.detect_btn.setEnabled(enable)
        self.measure_btn.setEnabled(enable)
        # self.save_btn.setEnabled(enable) # Save button enabled only after a test
        # self.reset_btn.setEnabled(enable) # Reset button enabled only after a test

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

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    # Ensure serial module is available
    try:
        import serial
    except ImportError:
        print("Python 'serial' module not found. Please install it using: pip install pyserial")
        sys.exit(1)

    app = QApplication(sys.argv)
    # Load custom fonts
    QFontDatabase.addApplicationFont("Montserrat-Black.ttf")
    QFontDatabase.addApplicationFont("Montserrat-Bold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-ExtraBold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-SemiBold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-Regular.ttf")

    ex = App()
    ex.show()
    sys.exit(app.exec_())




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

    def update_distance(self, distance):
        self.current_distance = distance
        self.diameter_label.setText(f"Wheel Diameter: {distance} mm")
        self.detect_btn.setVisible(False)
        self.measure_btn.setVisible(False)
        self.reset_btn.setVisible(True)
        self.save_btn.setVisible(True)
        self.diameter_label.show()

    def connect_signals(self):
        self.detect_btn.clicked.connect(self.detect_flaws)
        self.measure_btn.clicked.connect(self.measure_diameter)
        self.save_btn.clicked.connect(self.save_report)
        self.reset_btn.clicked.connect(self.reset_ui)

    def set_buttons_enabled(self, enabled):
        # Only enable measure button if we have a test result
        if hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"]:
            self.measure_btn.setEnabled(enabled)
        else:
            self.measure_btn.setEnabled(False)
        
        # Only enable save button if we have both test result and measurement
        if (hasattr(self, 'test_status') and self.test_status in ["FLAW DETECTED", "NO FLAW"] and self.current_distance != 680):
            self.save_btn.setEnabled(enabled)
        else:
            self.save_btn.setEnabled(False)

    def trigger_animation(self):
        self.status_animation.start()

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, status, recommendation):
        if status in ["FLAW DETECTED", "NO FLAW"]:
            if hasattr(self, 'current_distance'):
                self.diameter_label.setText("Wheel Diameter: Measure Next")
            else:
                self.diameter_label.setText("Wheel Diameter: -")
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
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: none;
                }
            """)
        
        self.trigger_animation()

    def detect_flaws(self):
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
        
        # Immediately disable the button for visual feedback
        self.detect_btn.setEnabled(False)
        self.measure_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.camera_thread.start_test()

    def measure_diameter(self):
        self.diameter_label.setText("Measuring...")
        self.diameter_label.show()

        self.detect_btn.setEnabled(False)
        self.measure_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        try:
            # VL53L0X is handled by Arduino, so this part is removed
            # self.tof = VL53L0X.VL53L0X()
            # self.tof.open()
            # time.sleep(0.1)
            # self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)

            self.sensor_thread = DistanceSensorThread()
            self.sensor_thread.distance_measured.connect(self.update_distance)
            self.sensor_thread.measurement_complete.connect(self.on_measurement_complete)
            self.sensor_thread.error_occurred.connect(self.on_measurement_error)
            self.sensor_thread.start()

        except Exception as e:
            print(f"Sensor init error: {e}")
            self.on_measurement_error("Sensor init error")

    def on_measurement_error(self, error_msg):
        print(f"Measurement error: {error_msg}")
        self.on_measurement_complete()

    def on_measurement_complete(self):
        # After measurement, show Reset and Save buttons
        self.detect_btn.setVisible(False)
        self.measure_btn.setVisible(False)
        self.save_btn.setEnabled(True)
        self.save_btn.setVisible(True)
        self.reset_btn.setVisible(True)

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
        self.detect_btn.setEnabled(False)
        self.detect_btn.setVisible(True)
        self.measure_btn.setEnabled(True)
        self.measure_btn.setVisible(True)
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)

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
                font-size: 14px;
            }
            QPushButton {
                background-color: #006600;
                color: white;
                border: none;
                padding: 8px 16px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #004400; }
            #qt_msgbox_buttonbox { border-top: 1px solid #ddd; padding-top: 16px; }
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
        
        # Reset buttons to initial state
        self.detect_btn.setEnabled(True)
        self.detect_btn.setVisible(True)
        self.measure_btn.setEnabled(False)
        self.measure_btn.setVisible(True)
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)

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