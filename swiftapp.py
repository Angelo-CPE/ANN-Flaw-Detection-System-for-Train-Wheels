import sys
import time
import cv2
import torch
import numpy as np
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QWidget, QFrame, QMessageBox, QSizePolicy)
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
    status_signal = pyqtSignal(str, str)
    countdown_signal = pyqtSignal(int)
    test_complete_signal = pyqtSignal(np.ndarray, str, str)
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
        model_path = 'ANN_model.pth'
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
        self._countdown = 3  # Changed to 3 to match the countdown sequence

    def run(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status_signal.emit("Error", "Cannot access camera")
                return

        self.status_signal.emit("Ready", "Press Start to begin")

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame.copy()
                processed_frame = frame.copy()
                
                if self._testing:
                    if self._countdown > 0:
                        # Emit countdown signal first before processing frame
                        self.countdown_signal.emit(self._countdown)
                        cv2.putText(processed_frame, f"{self._countdown}", 
                                    (processed_frame.shape[1]//2 - 20, processed_frame.shape[0]//2 + 20), 
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)
                        self._countdown -= 1
                        time.sleep(1)  # Consistent 1-second delay
                    elif self._countdown == 0:
                        self.countdown_signal.emit(-1)  # Signal for "IMAGE CAPTURED"
                        cv2.putText(processed_frame, "IMAGE CAPTURED", 
                                    (processed_frame.shape[1]//2 - 120, processed_frame.shape[0]//2 + 20), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        self._countdown = -1
                        time.sleep(1)
                    else:
                        self._testing = False
                        features = self.preprocess_image(frame)
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
                        self.test_complete_signal.emit(frame, status, recommendation)
                        self.animation_signal.emit()
                
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
# MAIN APPLICATION WINDOW
# ==============================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel Inspection")
        self.setWindowIcon(QIcon("icon.png"))
        self.setFixedSize(800, 480)
        
        self.load_fonts()
        
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
        
        logo_pixmap = QPixmap('D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/logo.png')
        if not logo_pixmap.isNull():
            self.logo_space.setPixmap(logo_pixmap.scaledToHeight(150, Qt.SmoothTransformation))
        
        self.status_title = QLabel("INSPECTION STATUS")
        self.status_title.setAlignment(Qt.AlignCenter)
        self.status_title.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat Black';
                font-size: 20px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
        """)
        
        self.status_indicator = QLabel("READY")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: black;
                font-family: 'Montserrat ExtraBold';
                font-size: 18px;
                padding: 15px 0;
            }
        """)
        
        self.recommendation_indicator = QLabel("Press Start to begin")
        self.recommendation_indicator.setAlignment(Qt.AlignCenter)
        self.recommendation_indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Montserrat';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        
        self.countdown_label = QLabel()
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            QLabel {
                color: red;
                font-family: 'Montserrat ExtraBold';
                font-size: 36px;
            }
        """)
        
        self.status_layout.addWidget(self.logo_space)
        self.status_layout.addWidget(self.status_title)
        self.status_layout.addWidget(self.status_indicator)
        self.status_layout.addWidget(self.recommendation_indicator)
        self.status_layout.addWidget(self.countdown_label)
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

    def load_fonts(self):
        font_db = QFontDatabase()
        
        regular_font_id = font_db.addApplicationFont("D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/Montserrat-Regular.ttf")
        extrabold_font_id = font_db.addApplicationFont("D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/Montserrat-ExtraBold.ttf")
        black_font_id = font_db.addApplicationFont("D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/Montserrat-Black.ttf")
        
        if regular_font_id == -1: print("Failed to load Montserrat-Regular")
        if extrabold_font_id == -1: print("Failed to load Montserrat-ExtraBold")
        if black_font_id == -1: print("Failed to load Montserrat-Black")
        
        QApplication.instance().setFont(QFont("Montserrat", 10))

    def setup_animations(self):
        self.status_animation = QPropertyAnimation(self.status_indicator, b"windowOpacity")
        self.status_animation.setDuration(300)
        self.status_animation.setStartValue(0.7)
        self.status_animation.setEndValue(1.0)

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

    def trigger_animation(self):
        self.status_animation.start()

    def update_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, status, recommendation):
        self.status_indicator.setText(status)
        self.recommendation_indicator.setText(recommendation)
        
        if "FLAW" in status:
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
                    color: red;
                    font-family: 'Montserrat';
                    font-size: 14px;
                    padding: 10px 0;
                }
            """)
            # Set camera border to red
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 4px solid red;
                }
            """)
        else:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: green;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 18px;
                    padding: 15px 0;
                }
            """)
            self.recommendation_indicator.setStyleSheet("""
                QLabel {
                    color: green;
                    font-family: 'Montserrat';
                    font-size: 14px;
                    padding: 10px 0;
                }
            """)
            # Set camera border to green
            self.camera_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 4px solid green;
                }
            """)
        
        self.trigger_animation()

    def update_countdown(self, count):
        if count > 0:
            self.countdown_label.setText(f"{count}")
            anim = QPropertyAnimation(self.countdown_label, b"geometry")
            anim.setDuration(200)
            anim.setEasingCurve(QEasingCurve.OutBack)
            anim.setStartValue(self.countdown_label.geometry())
            anim.setEndValue(self.countdown_label.geometry().adjusted(0, -5, 0, -5))
            anim.start()
        elif count == -1:
            self.countdown_label.setText("IMAGE CAPTURED")
            self.countdown_label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-family: 'Montserrat';
                    font-size: 14px;
                }
            """)
        else:
            self.countdown_label.setText("")
            self.countdown_label.setStyleSheet("""
                QLabel {
                    color: red;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 36px;
                }
            """)

    def start_test(self):
        self.start_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_indicator.setText("ANALYZING...")
        self.recommendation_indicator.setText("Processing wheel image")
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
                color: black;
                font-family: 'Montserrat';
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        # Reset camera border during analysis
        self.camera_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
            }
        """)
        self.camera_thread.start_test()

    def reset_app(self):
        self.start_btn.setText("START INSPECTION")
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status_indicator.setText("READY")
        self.recommendation_indicator.setText("Press Start to begin")
        self.countdown_label.setText("")
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
        self.countdown_label.setStyleSheet("""
            QLabel {
                color: red;
                font-family: 'Montserrat ExtraBold';
                font-size: 36px;
            }
        """)
        # Reset camera border to default
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
        self.save_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
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
            filename = f"wheel_inspection_{timestamp}.jpg"
            cv2.imwrite(filename, self.test_image)
            
            with open(f"wheel_inspection_{timestamp}.txt", "w") as f:
                f.write(f"Inspection Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {self.test_status}\n")
                f.write(f"Recommendation: {self.test_recommendation}\n")
            
            success_msg = QMessageBox()
            success_msg.setWindowTitle("Success")
            success_msg.setText(f"Saved as {filename}")
            success_msg.setIcon(QMessageBox.Information)
            success_msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                    border: 1px solid #ddd;
                    font-family: 'Montserrat';
                }
                QLabel { color: black; font-size: 14px; }
                QPushButton {
                    background-color: black;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    font-family: 'Montserrat ExtraBold';
                    font-size: 14px;
                    min-width: 80px;
                }
                QPushButton:hover { background-color: #333; }
            """)
            success_msg.exec_()
        
        self.save_btn.setEnabled(False)

    def closeEvent(self, event):
        self.camera_thread.stop()
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