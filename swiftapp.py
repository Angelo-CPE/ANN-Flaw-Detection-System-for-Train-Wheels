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
    sensor_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tof = None
        self.measurement_time = 1.0  # Measure for 1 second
        self.measurements = []

    def run_measurement(self):
        if not VL53L0X_AVAILABLE:
            # Simulate measurement if sensor not available
            self.measurements = [680] * 5  # Return 5 simulated measurements
            self.sensor_finished.emit()
            return True
            
        try:
            self.tof = VL53L0X.VL53L0X()
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)
            
            start_time = time.time()
            while time.time() - start_time < self.measurement_time and self._run_flag:
                distance = self.tof.get_distance()
                if distance > 0:
                    self.measurements.append(distance)
                    self.distance_measured.emit(distance)
                time.sleep(0.05)  # 20Hz measurement rate
            
            if not self.measurements:  # If no valid measurements
                self.measurements = [680]  # Default value
                
            self.sensor_finished.emit()
            return True
            
        except Exception as e:
            print(f"Error with distance sensor: {e}")
            self.measurements = [680]  # Default value
            self.sensor_finished.emit()
            return False
        finally:
            self.cleanup_sensor()

    def cleanup_sensor(self):
        if self.tof is not None:
            try:
                self.tof.stop_ranging()
                self.tof.close()
            except Exception as e:
                print(f"Error cleaning up sensor: {e}")
            finally:
                self.tof = None

    def stop(self):
        self._run_flag = False
        self.cleanup_sensor()
        self.wait()

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Reduced from 256
        self.dropout1 = nn.Dropout(0.2)        # Reduced from 0.3
        self.fc2 = nn.Linear(128, 64)          # Reduced from 128
        self.fc3 = nn.Linear(64, 2)            # Removed one layer

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
    distance_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._testing = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_frame = None
        self.current_distance = 680  # Default value
        self.load_model()
        
        # Optimized image size for processing
        self.processing_size = (96, 96)  # Reduced from 128x128

    def load_model(self):
        try:
            input_size = 2019
            self.model = ANNModel(input_size=input_size).to(self.device)
            model_path = 'ANN_model.pth'
            if os.path.exists(model_path):
                # Load with map_location to avoid CUDA memory issues
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                # Freeze model to reduce memory usage
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, frame):
        try:
            # Optimized HOG parameters
            frame_gray = cv2.cvtColor(cv2.resize(frame, self.processing_size), cv2.COLOR_BGR2GRAY)
            hog_features = hog(frame_gray, 
                             pixels_per_cell=(16, 16), 
                             cells_per_block=(1, 1),  # Reduced from 2x2
                             visualize=False)
            
            # Simplified signal processing
            signal = np.mean(frame_gray, axis=0)
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            
            # Combine features
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
        self.distance_requested.emit()  # Request distance measurement

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
        # Optimized GStreamer pipeline for Jetson Nano
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, "
            "format=NV12, framerate=15/1 ! "  # Reduced framerate
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=480, height=360, format=BGRx ! "  # Reduced resolution
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"  # Drop frames if processing is slow
        )
        
        cap = None
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("Failed to open with GStreamer, trying default camera")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

            self.status_signal.emit("Ready", "")

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    self.last_frame = frame.copy()
                    
                    # Only process if we're testing and have a frame
                    if self._testing:
                        self._testing = False
                        self.process_captured_image()
                    
                    # Convert to QImage
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
        
        # Camera Panel (simplified)
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
        
        # Control Panel (simplified)
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
        
        # Logo (simplified)
        self.logo_space = QLabel()
        self.logo_space.setAlignment(Qt.AlignCenter)
        self.logo_space.setFixedHeight(80)
        self.logo_space.setStyleSheet("background: transparent;")
        
        # Number controls (simplified)
        self.number_controls = QFrame()
        self.number_controls.setStyleSheet("QFrame { background: transparent; }")
        self.number_layout = QGridLayout()
        self.number_layout.setContentsMargins(0, 0, 0, 0)
        self.number_layout.setSpacing(5)
        
        # Simplified UI elements setup...
        # ... [previous UI setup code remains largely the same, but with reduced complexity]
        
        self.setup_animations()
        self.setup_camera_thread()
        self.setup_distance_sensor()
        self.connect_signals()
        self.setup_number_controls()

    def setup_distance_sensor(self):
        self.distance_sensor_thread = DistanceSensorThread()
        self.distance_sensor_thread.distance_measured.connect(self.update_distance)
        self.distance_sensor_thread.sensor_finished.connect(self.sensor_measurement_complete)
        self.camera_thread.distance_requested.connect(self.start_distance_measurement)

    def start_distance_measurement(self):
        # Start distance measurement in a separate thread to avoid blocking
        self.distance_sensor_thread.start()

    def sensor_measurement_complete(self):
        # Calculate median distance from measurements
        if self.distance_sensor_thread.measurements:
            self.current_distance = int(np.median(self.distance_sensor_thread.measurements))

    def update_distance(self, distance):
        # Update UI with live distance during measurement
        self.current_distance = distance
        self.diameter_label.setText(f"Measuring: {distance} mm")
        self.diameter_label.show()

    # ... [rest of the methods remain largely the same, with optimizations where needed]

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