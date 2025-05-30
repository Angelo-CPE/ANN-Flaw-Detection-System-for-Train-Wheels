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

# Import VL53L0X library
try:
    import VL53L0X
except ImportError:
    print("VL53L0X library not found. Distance measurement will be simulated.")
    VL53L0X = None

def send_report_to_backend(status, recommendation, image_base64, name=None, notes="", trainNumber=None, compartmentNumber=None, wheelNumber=None, wheel_diameter=None):
    import tempfile

    backend_url = "http://localhost:5000/api/reports"  # Change if hosted elsewhere

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
                'notes': notes,
                'trainNumber': str(trainNumber),
                'compartmentNumber': str(compartmentNumber),
                'wheelNumber': str(wheelNumber),
                'wheel_diameter': str(wheel_diameter)
            }

            response = requests.post(backend_url, files=files, data=data)

            if response.status_code == 201:
                print("Report sent successfully!")
            else:
                print(f"Failed to send report: {response.text}")
    except Exception as e:
        print(f"Error sending report: {e}")
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

class DistanceSensorThread(QThread):
    distance_measured = pyqtSignal(int)
    measurement_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tof = None

    def run(self):
        if VL53L0X is None:
            # Simulate measurement
            self.distance_measured.emit(680)
            time.sleep(1.0)
            self.measurement_complete.emit()
            return

        try:
            self.tof = VL53L0X.VL53L0X()
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)
            
            # Take measurement for 1 second
            start_time = time.time()
            while time.time() - start_time < 1.0 and self._run_flag:
                distance = self.tof.get_distance()
                if distance > 0:
                    self.distance_measured.emit(distance)
                time.sleep(0.05)  # 20Hz measurement rate
            
            self.measurement_complete.emit()
        except Exception as e:
            print(f"Error with distance sensor: {e}")
            self.distance_measured.emit(680)
            self.measurement_complete.emit()
        finally:
            if self.tof is not None:
                self.tof.stop_ranging()
                self.tof.close()

    def stop(self):
        self._run_flag = False
        self.wait()

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Reduced size for Jetson Nano
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
            input_size = 2019
            self.model = ANNModel(input_size=input_size).to(self.device)
            model_path = 'ANN_model.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                # Freeze model parameters
                for param in self.model.parameters():
                    param.requires_grad = False
                print("ANN model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, frame):
        try:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2GRAY)  # Reduced size
            hog_features = hog(frame_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            signal = np.mean(frame_gray, axis=0)
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            combined_features = np.concatenate([hog_features, amplitude_envelope])
            if len(combined_features) != 2019:
                combined_features = np.resize(combined_features, 2019)
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
        # Optimized GStreamer pipeline for Jetson Nano at 20 FPS
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, "
            "format=NV12, framerate=20/1 ! "  # Reduced to 20 FPS
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=480, height=360, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        
        cap = None
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("Failed to open with GStreamer, trying default camera")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 20)  # Reduced to 20 FPS
                
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

            self.status_signal.emit("Ready", "")

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
            self.status_signal.emit("Error", "Camera error")
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
        self.setWindowIcon(QIcon("icon.png"))
        self.setFixedSize(800, 480)
            
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.current_distance = 680  # Default value
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
        
        # Train Number controls...
        # [Previous number controls code remains the same]
        
        # Status indicators...
        # [Previous status indicators code remains the same]
        
        # Button Panel
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(20, 20, 20, 20)
        self.button_layout.setSpacing(15)
        
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
        
        self.button_layout.addWidget(self.detect_btn)
        self.button_layout.addWidget(self.measure_btn)
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
        self.connect_signals()
        self.setup_number_controls()

    def setup_number_controls(self):
        # [Previous number controls setup remains the same]
        pass

    def update_number(self, number_type, change):
        # [Previous number update code remains the same]
        pass

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
        self.diameter_label.show()
        self.measure_btn.setEnabled(True)  # Re-enable measure button after measurement
        self.save_btn.setEnabled(True)  # Enable save button after measurement

    def connect_signals(self):
        self.detect_btn.clicked.connect(self.detect_flaws)
        self.measure_btn.clicked.connect(self.measure_diameter)
        self.save_btn.clicked.connect(self.save_report)

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
                self.diameter_label.setText(f"Wheel Diameter: {self.current_distance} mm")
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
        
        # Disable all buttons during processing
        self.detect_btn.setEnabled(False)
        self.measure_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.camera_thread.start_test()

    def measure_diameter(self):
        self.status_indicator.setText("MEASURING DIAMETER...")
        self.detect_btn.setEnabled(False)
        self.measure_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # Initialize and start distance sensor
        self.sensor_thread = DistanceSensorThread()
        self.sensor_thread.distance_measured.connect(self.update_distance)
        self.sensor_thread.measurement_complete.connect(self.on_measurement_complete)
        self.sensor_thread.start()

    def on_measurement_complete(self):
        self.status_indicator.setText(self.test_status)
        self.detect_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

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
        
        # Reset UI after saving
        self.reset_ui()

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
        
        # Reset buttons
        self.detect_btn.setEnabled(True)
        self.measure_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # Reset data
        self.current_distance = 680
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None

    def handle_test_complete(self, image, status, recommendation):
        self.test_image = image
        self.test_status = status
        self.test_recommendation = recommendation
        
        # Enable measure button after flaw detection
        self.detect_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.save_btn.setEnabled(False)  # Still need measurement before saving

    def closeEvent(self, event):
        self.camera_thread.stop()
        if hasattr(self, 'sensor_thread'):
            self.sensor_thread.stop()
        event.accept()

if __name__ == "__main__":
    # Reduce memory usage by disabling unnecessary features
    os.environ["QT_QUICK_BACKEND"] = "software"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Simplified palette
    palette = app.palette()
    palette.setColor(palette.Window, QColor(255, 255, 255))
    palette.setColor(palette.WindowText, QColor(0, 0, 0))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = App()
    window.show()
    
    # Set process priority
    try:
        os.nice(10)  # Lower priority to avoid hogging resources
    except:
        pass
    
    sys.exit(app.exec_())