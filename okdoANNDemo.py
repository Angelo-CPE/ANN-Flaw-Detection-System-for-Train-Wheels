import time
import cv2
import torch
import numpy as np
from skimage.feature import hog
from scipy.signal import hilbert
import torch.nn as nn

# Set device for Jetson Nano
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
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

input_size = 2019
model = ANNModel(input_size=input_size).to(device)
model.load_state_dict(torch.load('D:/THESIS/ANN-Flaw-Detection-System-for-Train-Wheels/ANN_model.pth', map_location=device))
model.eval()

# Function to preprocess the image and extract features
def preprocess_image(frame):
    frame_gray = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    hog_features = hog(frame_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

    # Apply Local Mean Decomposition (LMD)
    signal = np.mean(frame_gray, axis=0)
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    frequency = np.pad(np.diff(phase) / (2.0 * np.pi), (0, 1), mode='constant')

    combined_features = np.concatenate([hog_features, amplitude_envelope, frequency])

    # Ensure feature length matches the model input size
    if len(combined_features) != input_size:
        combined_features = np.resize(combined_features, input_size)

    return combined_features

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

# Open webcam feed
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

start_time = time.time()
timer_duration = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    features = preprocess_image(frame)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)

    label = "Flawed" if predicted.item() == 1 else "Not Flawed"

    # Display text
    height, width, _ = frame.shape
    cv2.putText(frame, "Status: " + label, (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if label == "Flawed" else (255, 255, 255), 2)

    # Show the feed
    cv2.imshow("Webcam Feed", frame)

    if time.time() - start_time >= timer_duration or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
