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
                            QGridLayout, QStackedWidget, QSlider, QSpacerItem)
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
        # Exact margins/spacings to match design
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 40, 20, 20)
        self.layout.setSpacing(20)

        # Logo centered
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo = QPixmap('logo.png')
        self.logo_label.setPixmap(logo.scaledToHeight(120, Qt.SmoothTransformation))
        self.layout.addWidget(self.logo_label)

        # Add spacer to push buttons down
        self.layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Buttons container
        self.btn_container = QVBoxLayout()
        self.btn_container.setSpacing(15)

        # INSPECTION
        btn_inspect = QPushButton("INSPECTION")
        btn_inspect.setFixedHeight(60)
        btn_inspect.setStyleSheet("""
            QPushButton {
                background-color: #E60000;
                color: white;
                border-radius: 8px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
            }
            QPushButton:hover { background-color: #CC0000; }
            QPushButton:pressed { background-color: #B30000; }
        """)
        btn_inspect.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))
        self.btn_container.addWidget(btn_inspect)

        # CALIBRATION
        btn_cal = QPushButton("CALIBRATION")
        btn_cal.setFixedHeight(60)
        btn_cal.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 8px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
            }
            QPushButton:hover { background-color: #111111; }
            QPushButton:pressed { background-color: #000000; }
        """)
        btn_cal.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(3))
        self.btn_container.addWidget(btn_cal)

        self.layout.addLayout(self.btn_container)
        self.layout.addStretch(1)
        self.setLayout(self.layout)

class SelectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        # Back + Logo row
        hl = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setStyleSheet("""
            QPushButton { background: #F0F0F0; color: #333; border: 1px solid #DDD; border-radius: 5px; padding: 6px 12px; font-size: 14px; }
            QPushButton:hover { background: #E60000; color: white; border-color: #E60000; }
        """)
        btn_back.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(0))
        hl.addWidget(btn_back, alignment=Qt.AlignLeft)
        hl.addStretch(1)
        logo = QLabel()
        logo.setPixmap(QPixmap('logo.png').scaledToHeight(40, Qt.SmoothTransformation))
        hl.addWidget(logo, alignment=Qt.AlignRight)
        self.layout.addLayout(hl)

        # Title underline
        title = QLabel("SELECT INSPECTION DETAILS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel { font-family: 'Montserrat Bold'; font-size: 18px; color: #333; padding-bottom: 4px; border-bottom: 2px solid #E60000; }
        """)
        self.layout.addWidget(title)

        # Slider groups
        for label_text, maximum in [("Train Number", 20), ("Compartment Number", 8), ("Wheel Number", 8)]:
            vbox = QVBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size:16px; color:#555; font-family:'Montserrat SemiBold';")
            vbox.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, maximum)
            slider.setValue(1)
            slider.setStyleSheet("""
                QSlider::groove:horizontal { height:8px; background:#E0E0E0; border-radius:4px; }
                QSlider::handle:horizontal { width:24px; height:24px; background:#E60000; border-radius:12px; margin:-8px 0; }
            """)
            vbox.addWidget(slider)
            val = QLabel("1")
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet("font-size:20px; color:#E60000; font-family:'Montserrat Bold';")
            vbox.addWidget(val)
            slider.valueChanged.connect(lambda val, lbl=val: lbl.setText(str(val)))
            self.layout.addLayout(vbox)

        # START INSPECTION
        btn_inspect = """
            QPushButton {
                background-color: #e60000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Montserrat Bold';
                font-size: 18px;
                margin-top: 10px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
        """
        
        btn_start = QPushButton("START INSPECTION")
        btn_start.setFixedHeight(60)
        btn_start.setStyleSheet(btn_inspect.styleSheet().replace('#E60000', '#E60000').replace('INSPECTION','START'))
        btn_start.clicked.connect(lambda: [self.parent.setSelection(), self.parent.stacked_widget.setCurrentIndex(2)])
        self.layout.addWidget(btn_start)
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

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Top row: back + logo
        top = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setStyleSheet(""" /* same style as selection back */ """)
        btn_back.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))
        top.addWidget(btn_back)
        top.addStretch(1)
        logo = QLabel()
        logo.setPixmap(QPixmap('logo.png').scaledToHeight(40, Qt.SmoothTransformation))
        top.addWidget(logo)
        self.layout.addLayout(top)

        # Camera feed
        lbl_cam = QLabel()
        lbl_cam.setStyleSheet("background:black; border:none;")
        lbl_cam.setMinimumSize(480,360)
        lbl_cam.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(lbl_cam, stretch=1)
        self.parent.camera_label_widget = lbl_cam

        # Selection summary
        lbl_sum = QLabel()
        lbl_sum.setAlignment(Qt.AlignCenter)
        lbl_sum.setStyleSheet("font-size:16px; font-family:'Montserrat SemiBold'; color:#333;")
        self.layout.addWidget(lbl_sum)
        self.parent.summary_label = lbl_sum

        # INSPECTION STATUS
        lbl_stat = QLabel("INSPECTION STATUS")
        lbl_stat.setAlignment(Qt.AlignCenter)
        lbl_stat.setStyleSheet("font-size:18px; font-family:'Montserrat Bold'; color:#000;")
        self.layout.addWidget(lbl_stat)
        self.parent.title_status = lbl_stat

        lbl_val = QLabel("READY")
        lbl_val.setAlignment(Qt.AlignCenter)
        lbl_val.setStyleSheet("font-size:16px; font-family:'Montserrat ExtraBold'; color:#000;")
        self.layout.addWidget(lbl_val)
        self.parent.value_status = lbl_val

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
                color: #333;
                font-family: 'Montserrat Regular';
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
        
        # Button Panel - Modified for horizontal layout
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("QFrame { background: white; border: none; }")
        self.button_layout = QHBoxLayout()  # Changed to QHBoxLayout
        self.button_layout.setContentsMargins(0, 10, 0, 10)
        self.button_layout.setSpacing(10)
        
        # 1. Detect Flaws Button - Modified to be more square
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
            }
            QPushButton:hover { background-color: %s; }
            QPushButton:pressed { background-color: %s; }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """
        self.detect_btn = QPushButton("DETECT\nFLAWS")  # Added line break
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setStyleSheet(button_style % ("#e60000", "#cc0000", "#b30000"))
        
        # 2. Measure Diameter Button - Modified to be more square
        self.measure_btn = QPushButton("MEASURE\nDIAMETER")  # Added line break
        self.measure_btn.setEnabled(False)
        self.measure_btn.setCursor(Qt.PointingHandCursor)
        self.measure_btn.setStyleSheet(button_style % ("#0066cc", "#0055aa", "#004488"))
        
        # 3. Save Report Button - Modified to be more square
        self.save_btn = QPushButton("SAVE\nREPORT")  # Added line break
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet(button_style % ("#FFC107", "#FFB300", "#FFA000"))
        
        # 4. Reset Button - Modified to be more square
        self.reset_btn = QPushButton("NEW\nINSPECTION")  # Added line break
        self.reset_btn.setVisible(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(button_style % ("#000000", "#333333", "#222222"))
        
        self.button_layout.addWidget(self.detect_btn)
        self.button_layout.addWidget(self.measure_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.reset_btn)
        
        self.button_panel.setLayout(self.button_layout)
        self.control_layout.addWidget(self.button_panel)
        
        self.control_panel.setLayout(self.control_layout)
        self.layout.addWidget(self.control_panel, 30)  # Reduced weight to 30
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
                font-family: 'Montserrat Regular';
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
                font-family: 'Montserrat Regular';
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