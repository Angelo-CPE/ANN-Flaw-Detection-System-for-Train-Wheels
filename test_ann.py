import cv2
import torch
import numpy as np
import psutil
import time
from skimage.feature import hog
from scipy.signal import hilbert
from tqdm import tqdm
import torch.nn as nn

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


input_size = 2019  # Adjust based on feature size used in training
model = ANNModel(input_size=input_size)
model.load_state_dict(torch.load('/home/team39/Documents/ANN/ANN_model.pth'))
model.eval()

# Function to preprocess the image and extract features
def preprocess_image(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(frame_gray, (128, 128))

    # Extract HOG features
    hog_features = hog(resized_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

    # Apply Local Mean Decomposition (LMD)
    signal = np.mean(resized_image, axis=0)
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    frequency = np.diff(phase) / (2.0 * np.pi)

    combined_features = np.concatenate([hog_features, amplitude_envelope, frequency])

    return combined_features

# Metrics trackers
latency_list = []
memory_usage_list = []
start_time = time.time()

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Timer setup
timer_duration = 60  # 1 minute
while True:
    # Start time for latency measurement
    frame_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Preprocess the frame and make predictions
    features = preprocess_image(frame)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    outputs = model(features_tensor)
    _, predicted = torch.max(outputs, 1)

    label = "Flawed" if predicted.item() == 1 else "Not Flawed"

    # End time for latency measurement
    frame_end_time = time.time()
    latency_list.append(frame_end_time - frame_start_time)

    # Calculate memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    memory_usage_list.append(memory_usage)

    # Display prediction on the video feed
    cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Avg Latency: {np.mean(latency_list) * 1000:.2f} ms", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory Usage: {np.mean(memory_usage_list):.2f} MB", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam Feed", frame)

    # Check if the timer has exceeded 1 minute
    elapsed_time = time.time() - start_time
    if elapsed_time >= timer_duration:
        break

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Calculate total runtime
end_time = time.time()
runtime_seconds = end_time - start_time

# Display final metrics
print(f"Average Latency: {np.mean(latency_list) * 1000:.2f} ms")
print(f"Average Memory Usage: {np.mean(memory_usage_list):.2f} MB")
print(f"Total Runtime: {runtime_seconds:.2f} seconds")

# Estimate power consumption (approximation, may vary by system)
power_consumption = runtime_seconds * 0.01
print(f"Estimated Power Consumption: {power_consumption:.2f} Watts")

