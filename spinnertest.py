import sys
import time
import os
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt

class SpinnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spinner Demo")
        self.setFixedSize(200, 200)

        layout = QVBoxLayout(self)

        self.label = QLabel("READY", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.spinner_label = QLabel(self)
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setFixedSize(200, 200)
        layout.addWidget(self.spinner_label)

        # Use spinner.gif file located in the same directory
        spinner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spinner.gif")
        self.spinner = QMovie(spinner_path)
        self.spinner.setScaledSize(self.spinner_label.size())
        self.spinner_label.setMovie(self.spinner)
        self.spinner_label.hide()

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_spinner)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_spinner)
        layout.addWidget(self.stop_btn)

    def start_spinner(self):
        self.label.setText("Analyzing")
        self.spinner.start()
        self.spinner_label.show()

    def stop_spinner(self):
        self.label.setText("Done")
        self.spinner.stop()
        self.spinner_label.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpinnerApp()
    window.show()
    sys.exit(app.exec_())
