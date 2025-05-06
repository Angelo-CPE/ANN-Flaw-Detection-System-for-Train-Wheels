import time
import cv2
import torch
import numpy as np
from skimage.feature import hog
from scipy.signal import hilbert
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

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Timer setup
timer_duration = 30  # 1 minute
start_time = time.time()  # Initialize the start time before the loop

while True:
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

    # Define the status text and colors
    status_label = "Status:"
    recommendation_label = "Recommendation: "
    if label == "Flawed":
        status_text = "Flawed"
        recommendation_text = "     For Replacement"
        status_color = (0, 0, 255)  # Red
        recommendation_color = (0, 0, 255)  # Red
    else:
        status_text = "Not Flawed"
        recommendation_text =  "     For Consistent Monitoring"
        status_color = (255, 255, 255)  # White (default)
        recommendation_color = (255, 255, 255)  # White (default)

    # Define text positions for the bottom left
    height, width, _ = frame.shape
    start_x = 10
    start_y = height - 60  # Adjust for bottom-left positioning

    # Display the text on the frame
    cv2.putText(frame, "Artificial Neural Network Model", (start_x, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (start_x + 100, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, recommendation_label, (start_x, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, recommendation_text, (start_x + 150, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recommendation_color, 2)

    # Show the webcam feed with annotations
    cv2.imshow("Webcam Feed", frame)

    # Check if the timer has exceeded 1 minute
    elapsed_time = time.time() - start_time
    if elapsed_time >= timer_duration:
        print("1 minute has passed, stopping the feed.")
        break

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
