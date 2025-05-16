import sys
import time
import cv2
import torch
import numpy as np
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QFrame, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QFontDatabase, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QPropertyAnimation, QEasingCurve

# ==============================================
# MODEL DEFINITION
# ==============================================
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

# ==============================================
# CAMERA THREAD
# ==============================================
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str)  # (status, recommendation)
    countdown_signal = pyqtSignal(int)
    test_complete_signal = pyqtSignal(np.ndarray, str, str)  # (image, status, recommendation)
    animation_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._testing = False
        self._countdown = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_frame = None
        self.load_model()

    def load_model(self):
        input_size = 2019
        self.model = ANNModel(input_size=input_size).to(self.device)
        model_path = 'ANN_model.pth'  # Update this path to your model file
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess_image(self, frame):
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

    def start_test(self):
        self._testing = True
        self._countdown = 2  # 2 second countdown

    def run(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

        self.status_signal.emit("Ready", "Press 'Start Test' to begin")

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame.copy()
                processed_frame = frame.copy()
                
                # Display countdown if testing
                if self._testing:
                    if self._countdown > 0:
                        # Show countdown on frame with better styling
                        cv2.putText(processed_frame, f"{self._countdown}", 
                                    (processed_frame.shape[1]//2 - 30, processed_frame.shape[0]//2 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
                        self.countdown_signal.emit(self._countdown)
                        self._countdown -= 1
                        time.sleep(1)
                    else:
                        # Capture and process frame
                        self._testing = False
                        features = self.preprocess_image(frame)
                        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(features_tensor)
                            _, predicted = torch.max(outputs, 1)
                        
                        if predicted.item() == 1:
                            status = "Flawed"
                            recommendation = "For Replacement"
                            color = (0, 0, 255)  # Red
                        else:
                            status = "Not Flawed"
                            recommendation = "In good condition"
                            color = (0, 255, 0)  # Green
                        
                        # Add result to frame with better styling
                        cv2.putText(processed_frame, f"Status: {status}", (20, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                        cv2.putText(processed_frame, f"Recommendation: {recommendation}", (20, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                        cv2.rectangle(processed_frame, (10, 10), (processed_frame.shape[1]-10, 80), color, 2)
                        
                        self.status_signal.emit(status, recommendation)
                        self.test_complete_signal.emit(frame, status, recommendation)
                        self.animation_signal.emit()
                
                # Convert to QImage
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self._run_flag = False
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

# ==============================================
# MAIN APPLICATION WINDOW (Responsive UI)
# ==============================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Wheel Inspection")
        self.setWindowIcon(QIcon("icon.png"))  # Add your icon file
        self.setGeometry(0, 0, 800, 480)
        
        # Load custom font
        font_id = QFontDatabase.addApplicationFont("D:\THESIS\ANN-Flaw-Detection-System-for-Train-Wheels\poppins.regular.ttf")
        if font_id != -1:
            self.font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            self.font_family = "Segoe UI"  # Modern system font fallback
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                       stop:0 #f5f5f5, stop:1 #e0e0e0);
        """)
        
        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.central_widget.setLayout(self.main_layout)
        
        # Camera View with card-like design
        self.camera_frame = QFrame()
        self.camera_frame.setFrameShape(QFrame.StyledPanel)
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setContentsMargins(5, 5, 5, 5)
        
        # Camera label with shadow effect
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }
        """)
        
        # Add overlay title with modern styling
        self.camera_title = QLabel("LIVE INSPECTION FEED")
        self.camera_title.setAlignment(Qt.AlignCenter)
        self.camera_title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 12px;
                font-weight: 600;
                padding: 8px 15px;
                background-color: #e74c3c;
                border-radius: 6px;
                border: none;
            }
        """)
        
        self.camera_layout.addWidget(self.camera_title, 0, Qt.AlignHCenter)
        self.camera_layout.addSpacing(5)
        self.camera_layout.addWidget(self.camera_label)
        self.camera_frame.setLayout(self.camera_layout)
        self.camera_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: none;
            }
        """)
        
        # Controls Frame
        self.controls_frame = QFrame()
        self.controls_frame.setFrameShape(QFrame.StyledPanel)
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(15)
        
        # Status Panel with improved layout
        self.status_group = QFrame()
        self.status_group.setFrameShape(QFrame.StyledPanel)
        self.status_layout = QVBoxLayout()
        self.status_layout.setContentsMargins(10, 10, 10, 10)
        self.status_layout.setSpacing(8)
        
        self.status_title = QLabel("STATUS")
        self.status_title.setAlignment(Qt.AlignCenter)
        self.status_title.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: 600;
                padding: 6px;
                background-color: #3498db;
                border-radius: 6px;
                border: none;
                font-size: 12px;
            }
        """)
        
        # Status indicators with better spacing
        self.status_indicator = QLabel()
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setMinimumHeight(40)
        self.status_indicator.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        
        self.recommendation_indicator = QLabel()
        self.recommendation_indicator.setAlignment(Qt.AlignCenter)
        self.recommendation_indicator.setMinimumHeight(40)
        self.recommendation_indicator.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
                font-size: 12px;
            }
        """)
        
        self.countdown_label = QLabel()
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setMinimumHeight(40)
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #e74c3c;
            }
        """)
        
        # Add status widgets with proper spacing
        self.status_layout.addWidget(self.status_title)
        self.status_layout.addWidget(self.status_indicator)
        self.status_layout.addWidget(self.recommendation_indicator)
        self.status_layout.addWidget(self.countdown_label)
        self.status_group.setLayout(self.status_layout)
        self.status_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: none;
            }
        """)
        
        # Test Controls with modern buttons
        self.button_group = QFrame()
        self.button_group.setFrameShape(QFrame.StyledPanel)
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(10, 10, 10, 10)
        self.button_layout.setSpacing(15)
        
        self.start_btn = QPushButton("START TEST")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn = QPushButton("SAVE")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        
        # Modern button styling with transitions
        button_style = """
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
        """
        save_button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a6ca8;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
        """
        self.start_btn.setStyleSheet(button_style)
        self.save_btn.setStyleSheet(save_button_style)
        
        # Add control widgets with proper spacing
        self.button_layout.addWidget(self.start_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_group.setLayout(self.button_layout)
        self.button_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: none;
            }
        """)
        
        # Add widgets to controls frame
        self.controls_layout.addWidget(self.status_group)
        self.controls_layout.addWidget(self.button_group)
        self.controls_frame.setLayout(self.controls_layout)
        self.controls_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: none;
            }
        """)
        
        # Combine Main Layout
        self.main_layout.addWidget(self.camera_frame, 75)
        self.main_layout.addWidget(self.controls_frame, 25)
        
        # Setup animations
        self.setup_animations()
        
        # Initialize camera thread
        self.setup_camera_thread()

        # Connect signals
        self.connect_signals()

    def setup_animations(self):
        # Pulse animation for camera feed
        self.camera_animation = QPropertyAnimation(self.camera_label, b"geometry")
        self.camera_animation.setDuration(500)
        self.camera_animation.setEasingCurve(QEasingCurve.OutQuint)
        
        # Button press animation
        self.btn_animation = QPropertyAnimation(self.start_btn, b"geometry")
        self.btn_animation.setDuration(200)
        self.btn_animation.setEasingCurve(QEasingCurve.OutBack)
        
        # Status indicator animation
        self.status_animation = QPropertyAnimation(self.status_indicator, b"geometry")
        self.status_animation.setDuration(300)
        self.status_animation.setEasingCurve(QEasingCurve.OutBack)

    def setup_camera_thread(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.countdown_signal.connect(self.update_countdown)
        self.camera_thread.test_complete_signal.connect(self.handle_test_complete)
        self.camera_thread.animation_signal.connect(self.trigger_animation)
        self.camera_thread.start()

    def connect_signals(self):
        self.start_btn.clicked.connect(self.start_test)
        self.save_btn.clicked.connect(self.save_results)

    def trigger_animation(self):
        # Animate camera label with more subtle effect
        original_geometry = self.camera_label.geometry()
        self.camera_animation.setStartValue(original_geometry)
        self.camera_animation.setEndValue(original_geometry.adjusted(-2, -2, 2, 2))
        self.camera_animation.start()
        
        # Animate status indicator
        original_status_geo = self.status_indicator.geometry()
        self.status_animation.setStartValue(original_status_geo)
        self.status_animation.setEndValue(original_status_geo.adjusted(0, -3, 0, -3))
        self.status_animation.start()

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, status, recommendation):
        self.status_indicator.setText(f"Status: {status}")
        self.recommendation_indicator.setText(f"Rec: {recommendation}")
        
        if "Flawed" in status:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #ffebee;
                    border-radius: 6px;
                    border: 2px solid #ef9a9a;
                    font-size: 14px;
                    font-weight: 500;
                    color: #c62828;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    background-color: #ffebee;
                    border-radius: 6px;
                    border: 2px solid #ef9a9a;
                    font-size: 12px;
                    color: #c62828;
                }
            """)
        else:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    border-radius: 6px;
                    border: 2px solid #a5d6a7;
                    font-size: 14px;
                    font-weight: 500;
                    color: #2e7d32;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    border-radius: 6px;
                    border: 2px solid #a5d6a7;
                    font-size: 12px;
                    color: #2e7d32;
                }
            """)
        
        self.trigger_animation()

    def update_countdown(self, count):
        if count > 0:
            self.countdown_label.setText(f"{count}")
            self.countdown_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    color: #e74c3c;
                    background-color: #f5f5f5;
                    border-radius: 6px;
                }
            """)
            # Animate the countdown
            original_geo = self.countdown_label.geometry()
            anim = QPropertyAnimation(self.countdown_label, b"geometry")
            anim.setDuration(200)
            anim.setEasingCurve(QEasingCurve.OutBack)
            anim.setStartValue(original_geo)
            anim.setEndValue(original_geo.adjusted(0, -5, 0, -5))
            anim.start()
        else:
            self.countdown_label.setText("")
            self.countdown_label.setStyleSheet("")

    def start_test(self):
        # Animate button press
        original_geo = self.start_btn.geometry()
        self.btn_animation.setStartValue(original_geo)
        self.btn_animation.setEndValue(original_geo.adjusted(0, 3, 0, 3))
        self.btn_animation.start()
        
        self.start_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_indicator.setText("Status: Testing...")
        self.recommendation_indicator.setText("Processing...")
        self.status_indicator.setStyleSheet("""
            QLabel {
                background-color: #fff3e0;
                border-radius: 6px;
                border: 2px solid #ffcc80;
                font-size: 14px;
                font-weight: 500;
                color: #e65100;
            }
        """)
        self.recommendation_indicator.setStyleSheet("""
            QLabel {
                background-color: #fff3e0;
                border-radius: 6px;
                border: 2px solid #ffcc80;
                font-size: 12px;
                color: #e65100;
            }
        """)
        self.camera_thread.start_test()

    def handle_test_complete(self, image, status, recommendation):
        self.test_image = image
        self.test_status = status
        self.test_recommendation = recommendation
        self.save_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.trigger_animation()

    def save_results(self):
        # Create a styled message box
        msg = QMessageBox()
        msg.setWindowTitle("Save Results")
        msg.setText("Save this test result?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        # Style the message box
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                font-size: 12px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: 500;
                min-width: 60px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        reply = msg.exec_()
        
        if reply == QMessageBox.Yes:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"wheel_inspection_{timestamp}.jpg"
            cv2.imwrite(filename, self.test_image)
            
            # Save text file with results
            with open(f"wheel_inspection_{timestamp}.txt", "w") as f:
                f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {self.test_status}\n")
                f.write(f"Recommendation: {self.test_recommendation}\n")
            
            # Show success notification
            success_msg = QMessageBox()
            success_msg.setWindowTitle("Success")
            success_msg.setText(f"Saved as:\n{filename}")
            success_msg.setIcon(QMessageBox.Information)
            success_msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                    border: 2px solid #2ecc71;
                    border-radius: 8px;
                    font-size: 12px;
                }
                QLabel {
                    color: #2c3e50;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    min-width: 60px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """)
            success_msg.exec_()
        
        self.save_btn.setEnabled(False)

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style to Fusion for modern look
    app.setStyle("Fusion")
    
    # Create and show main window
    window = App()
    window.show()
    
    sys.exit(app.exec_())