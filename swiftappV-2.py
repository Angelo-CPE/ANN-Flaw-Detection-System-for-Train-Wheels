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
            files = {'image': ('inspection.jpg', img_file, 'image/jpeg')}
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
                return response.status_code == 201
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")
                return False
                
    except Exception as e:
        print(f"Error preparing report: {e}")
        return False
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

class BatteryMonitorThread(QThread):
    battery_percentage_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.ina219 = INA219()

    def run(self):
        while self._run_flag:
            try:
                bus_voltage = self.ina219.getBusVoltage_V()
                percentage = ((bus_voltage - 7.0) / (8.4 - 7.0)) * 100
                percentage = max(0, min(100, int(percentage)))
                self.battery_percentage_signal.emit(percentage)
            except Exception as e:
                print(f"Error reading battery data: {e}")
                self.battery_percentage_signal.emit(-1)
            time.sleep(5)

    def stop(self):
        self._run_flag = False
        self.wait()

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
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                print(f"Connected to {self.port} at {self.baudrate} baud")
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                time.sleep(2)
                
                while ser.in_waiting:
                    ser.readline()

                ser.write(b'R\n')
                
                start_time = time.time()
                while time.time() - start_time < 5:
                    if ser.in_waiting:
                        line = ser.readline().decode('ascii', errors='ignore').strip()
                        if line:
                            print(f"Received: {line}")
                            try:
                                distance = int(line)
                                if distance > 0:
                                    self.distance_measured.emit(distance)
                                    self.measurement_complete.emit()
                                    return
                                elif distance == 0:
                                    self.error_occurred.emit("No new measurement available")
                                    break
                                elif distance == -1:
                                    self.error_occurred.emit("Sensor initialization error")
                                    break
                                elif distance == -2:
                                    self.error_occurred.emit("Sensor timeout")
                                    break
                            except ValueError:
                                print(f"Non-integer data: {line}")
                                continue
                
                self.error_occurred.emit("No valid measurement received")
                
        except serial.SerialException as e:
            self.error_occurred.emit(f"Serial port error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {str(e)}")
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
                print("ANN model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

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
            print(f"Image preprocessing error: {e}")
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
                print("Falling back to USB camera...")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 20)

            if not cap.isOpened():
                print("Error: Cannot access any camera")
                return

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    self.last_frame = frame.copy()
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image)

        except Exception as e:
            print(f"Camera error: {e}")
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
        self.current_distance = 680
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None
        
        self.setup_ui()
        self.setup_animations()
        self.setup_camera_thread()
        self.connect_signals()

        # Battery Monitor Thread
        self.battery_monitor_thread = BatteryMonitorThread()
        self.battery_monitor_thread.battery_percentage_signal.connect(self.update_battery_percentage)
        self.battery_monitor_thread.start()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background: white;")
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setLayout(self.main_layout)
        
        self.content_layout = QHBoxLayout()
        
        # Camera Panel
        self.camera_panel = QFrame()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet("QLabel { background: black; }")
        
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(self.camera_label)
        self.camera_panel.setLayout(camera_layout)
        
        # Control Panel
        self.control_panel = QFrame()
        control_layout = QVBoxLayout()
        
        # Status Panel
        self.status_panel = QFrame()
        status_layout = QVBoxLayout()
        
        # Logo
        self.logo_space = QLabel()
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            self.logo_space.setPixmap(logo_pixmap.scaledToHeight(150, Qt.SmoothTransformation))
        
        # Number controls
        self.setup_number_controls()
        
        # Status indicators
        self.status_title = QLabel("INSPECTION STATUS")
        self.status_indicator = QLabel("READY")
        self.analyzing_label = QLabel("ANALYZING")
        self.analyzing_label.hide()
        self.recommendation_indicator = QLabel()
        self.diameter_label = QLabel("Wheel Diameter: -")
        self.diameter_label.hide()
        self.battery_label = QLabel("Battery: --%")
        
        # Add widgets to status layout
        status_layout.addWidget(self.logo_space)
        status_layout.addWidget(self.number_controls)
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.analyzing_label)
        status_layout.addWidget(self.recommendation_indicator)
        status_layout.addWidget(self.battery_label)
        status_layout.addWidget(self.diameter_label)
        self.status_panel.setLayout(status_layout)
        
        # Button Panel
        self.setup_buttons()
        
        # Add panels to control layout
        control_layout.addWidget(self.status_panel)
        control_layout.addWidget(self.button_panel)
        self.control_panel.setLayout(control_layout)
        
        # Add to main layout
        self.content_layout.addWidget(self.camera_panel)
        self.content_layout.addWidget(self.control_panel)
        self.main_layout.addLayout(self.content_layout)

    def setup_number_controls(self):
        self.number_controls = QFrame()
        layout = QGridLayout()
        
        # Train number controls
        self.train_label = QLabel("Train")
        self.train_decrement = QPushButton("-")
        self.trainNumber_label = QLabel("1")
        self.train_increment = QPushButton("+")
        
        # Compartment number controls
        self.compartment_label = QLabel("Compartment")
        self.compartment_decrement = QPushButton("-")
        self.compartmentNumber_label = QLabel("1")
        self.compartment_increment = QPushButton("+")
        
        # Wheel number controls
        self.wheel_label = QLabel("Wheel")
        self.wheel_decrement = QPushButton("-")
        self.wheelNumber_label = QLabel("1")
        self.wheel_increment = QPushButton("+")
        
        # Add to layout
        layout.addWidget(self.train_label, 0, 0)
        layout.addWidget(self.train_decrement, 0, 1)
        layout.addWidget(self.trainNumber_label, 0, 2)
        layout.addWidget(self.train_increment, 0, 3)
        
        layout.addWidget(self.compartment_label, 1, 0)
        layout.addWidget(self.compartment_decrement, 1, 1)
        layout.addWidget(self.compartmentNumber_label, 1, 2)
        layout.addWidget(self.compartment_increment, 1, 3)
        
        layout.addWidget(self.wheel_label, 2, 0)
        layout.addWidget(self.wheel_decrement, 2, 1)
        layout.addWidget(self.wheelNumber_label, 2, 2)
        layout.addWidget(self.wheel_increment, 2, 3)
        
        self.number_controls.setLayout(layout)

    def setup_buttons(self):
        self.button_panel = QFrame()
        layout = QVBoxLayout()
        
        self.detect_btn = QPushButton("DETECT FLAWS")
        self.measure_btn = QPushButton("MEASURE DIAMETER")
        self.save_btn = QPushButton("SAVE REPORT")
        self.reset_btn = QPushButton("RESET")
        
        # Style buttons
        button_style = """
            QPushButton {
                padding: 12px;
                font-family: 'Montserrat ExtraBold';
                font-size: 14px;
                border-radius: 4px;
                margin: 5px;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """
        
        self.detect_btn.setStyleSheet(button_style + """
            QPushButton { background-color: #e60000; color: white; }
            QPushButton:hover { background-color: #cc0000; }
        """)
        
        self.measure_btn.setStyleSheet(button_style + """
            QPushButton { background-color: #333; color: white; }
            QPushButton:hover { background-color: #111; }
        """)
        
        self.save_btn.setStyleSheet(button_style + """
            QPushButton { background-color: #006600; color: white; }
            QPushButton:hover { background-color: #004400; }
        """)
        self.save_btn.setVisible(False)
        
        self.reset_btn.setStyleSheet(button_style + """
            QPushButton { background-color: #444; color: white; }
            QPushButton:hover { background-color: #222; }
        """)
        self.reset_btn.setVisible(False)
        
        layout.addWidget(self.detect_btn)
        layout.addWidget(self.measure_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.reset_btn)
        self.button_panel.setLayout(layout)

    def setup_animations(self):
        self.status_animation = QPropertyAnimation(self.status_indicator, b"windowOpacity")
        self.status_animation.setDuration(300)
        self.status_animation.setStartValue(0.7)
        self.status_animation.setEndValue(1.0)

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
        diameter_calculated = 920 + (680 - self.current_distance) * 0.5
        self.diameter_label.setText(f"Wheel Diameter: {diameter_calculated:.2f} mm")
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
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, status, recommendation):
        self.status_indicator.setText(status)
        self.recommendation_indicator.setText(recommendation)
        
        if status == "FLAW DETECTED":
            self.status_indicator.setStyleSheet("color: red; font-family: 'Montserrat ExtraBold'; font-size: 18px;")
            self.camera_label.setStyleSheet("QLabel { background: black; border: 4px solid red; }")
            self.save_btn.setVisible(True)
            self.reset_btn.setVisible(True)
        elif status == "NO FLAW":
            self.status_indicator.setStyleSheet("color: #00CC00; font-family: 'Montserrat ExtraBold'; font-size: 18px;")
            self.camera_label.setStyleSheet("QLabel { background: black; border: 4px solid #00CC00; }")
            self.save_btn.setVisible(True)
            self.reset_btn.setVisible(True)
        else:
            self.status_indicator.setStyleSheet("color: black; font-family: 'Montserrat ExtraBold'; font-size: 18px;")
            self.camera_label.setStyleSheet("QLabel { background: black; border: none; }")
        
        self.trigger_animation()

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
        self.animation.setEndValue(current_rect.translated(10, 0))
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
        if self.test_image is None or self.test_status is None or self.test_recommendation is None:
            QMessageBox.warning(self, "Report", "No test data to save.")
            return

        msg = QMessageBox()
        msg.setWindowTitle("Save Report")
        msg.setText("Save this inspection report?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        
        if msg.exec_() == QMessageBox.Save:
            try:
                _, buffer = cv2.imencode('.jpg', self.test_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                report_name = f"Train {self.trainNumber} - Compartment {self.compartmentNumber} - Wheel {self.wheelNumber}"
                
                success = send_report_to_backend(
                    self.test_status,
                    self.test_recommendation,
                    image_base64,
                    name=report_name,
                    trainNumber=self.trainNumber,
                    compartmentNumber=self.compartmentNumber,
                    wheelNumber=self.wheelNumber,
                    wheel_diameter=self.current_distance
                )
                
                if success:
                    self.reset_application()
                    QMessageBox.information(self, "Success", "Report saved successfully!")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to save report.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

    def reset_application(self):
        self.trainNumber = 1
        self.compartmentNumber = 1
        self.wheelNumber = 1
        self.trainNumber_label.setText("1")
        self.compartmentNumber_label.setText("1")
        self.wheelNumber_label.setText("1")
        self.status_indicator.setText("READY")
        self.recommendation_indicator.setText("")
        self.diameter_label.hide()
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.enable_buttons(True)
        self.current_distance = 680
        self.test_image = None
        self.test_status = None
        self.test_recommendation = None

    def update_battery_percentage(self, percentage):
        if percentage == -1:
            self.battery_label.setText("Battery: Error")
        else:
            self.battery_label.setText(f"Battery: {percentage}%")

    def closeEvent(self, event):
        self.camera_thread.stop()
        if hasattr(self, 'distance_thread'):
            self.distance_thread.stop()
        if hasattr(self, 'battery_monitor_thread'):
            self.battery_monitor_thread.stop()
        event.accept()

if __name__ == "__main__":
    os.environ["QT_QUICK_BACKEND"] = "software"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load custom fonts
    QFontDatabase.addApplicationFont("Montserrat-Black.ttf")
    QFontDatabase.addApplicationFont("Montserrat-Bold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-ExtraBold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-SemiBold.ttf")
    QFontDatabase.addApplicationFont("Montserrat-Regular.ttf")

    window = App()
    window.show()
    
    try:
        os.nice(10)
    except:
        pass
    
    sys.exit(app.exec_())