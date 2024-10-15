import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import subprocess
import os
import time

# ----------------------------- Configuration -----------------------------

# Paths to the saved model, scaler, and label encoder
MODEL_SAVE_PATH = 'gesture_model.keras'       # Ensure this matches your saved model's extension
SCALER_SAVE_PATH = 'scaler.save'
LABEL_ENCODER_PATH = 'label_encoder.save'

# Verify that all required files exist
for path in [MODEL_SAVE_PATH, SCALER_SAVE_PATH, LABEL_ENCODER_PATH]:
    if not os.path.exists(path):
        print(f"Error: Required file '{path}' not found.")
        exit()

# Load label encoder to get gesture classes
le = joblib.load(LABEL_ENCODER_PATH)
gesture_classes = le.classes_
print(f"Gesture classes: {gesture_classes}")

# MediaPipe Hands Configuration
MAX_NUM_HANDS = 1          # Number of hands to detect
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# Gesture-Action Mapping
# Define which gesture corresponds to which system command
# Adjust the commands based on your operating system and desired applications

# Example for macOS
import platform

system = platform.system()

if system == 'Darwin':  # macOS
    GESTURE_ACTION_MAP = {
        'Wave': 'open -a "Safari"',
        'Point': 'open -a "Discord"',
        'Peace': 'open -a "Calendar"',
        'Thumbs Up': 'open -a "Messages"',
        'Okay': 'open -a "Maps"',
        # Add more macOS commands
    }
elif system == 'Windows':
    GESTURE_ACTION_MAP = {
        'Wave': 'start notepad.exe',
        # Add more Windows commands
    }
elif system == 'Linux':
    GESTURE_ACTION_MAP = {
        'Wave': 'gedit',
        # Add more Linux commands
    }
else:
    print("Unsupported Operating System.")
    exit()

# Confidence threshold to trigger actions
CONFIDENCE_THRESHOLD = 1

# Cooldown period in seconds to prevent multiple triggers
COOLDOWN_PERIOD = 10
last_trigger_time = {gesture: 0 for gesture in GESTURE_ACTION_MAP.keys()}

# --------------------------- Load the Model ------------------------------

# Load the trained TensorFlow/Keras model
model = load_model(MODEL_SAVE_PATH)
print(f"Model loaded successfully from '{MODEL_SAVE_PATH}'")

# Load the scaler
scaler = joblib.load(SCALER_SAVE_PATH)
print(f"Scaler loaded successfully from '{SCALER_SAVE_PATH}'")

# ---------------------- Initialize MediaPipe Hands -----------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,            # Video stream
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

# ------------------------ Helper Functions -------------------------------

def extract_hand_landmarks(image, hands_detector):
    """
    Processes the image and extracts hand landmarks.

    Args:
        image (numpy.ndarray): The input image.
        hands_detector: The MediaPipe Hands detector.

    Returns:
        list or None: Flattened list of hand landmarks or None if no hand is detected.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks  # Return the first detected hand's landmarks
    return None

def preprocess_landmarks(landmarks, scaler):
    """
    Preprocesses the landmarks to match the model's input requirements.

    Args:
        landmarks (list): The list of hand landmark coordinates.
        scaler (StandardScaler): The fitted scaler.

    Returns:
        numpy.ndarray: The preprocessed landmarks ready for prediction.
    """
    # Convert to numpy array
    landmarks = np.array(landmarks)

    # Reshape based on the model's input shape
    landmarks = landmarks.reshape(1, -1)

    # Scale the features
    landmarks = scaler.transform(landmarks)

    return landmarks

def execute_command(command):
    """
    Executes a system command.

    Args:
        command (str): The system command to execute.
    """
    try:
        subprocess.Popen(command, shell=True)
        print(f"Executed command: {command}")
    except Exception as e:
        print(f"Failed to execute command '{command}'. Error: {e}")

# -------------------- Real-Time Gesture Recognition ----------------------

def main():
    global last_trigger_time
    """
    Main function to perform real-time gesture recognition.
    """
    # Start video capture from the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Real-Time Gesture Recognition. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally for a mirror effect (optional)
        frame = cv2.flip(frame, 1)

        # Extract hand landmarks
        landmarks = extract_hand_landmarks(frame, hands)

        gesture = "No Gesture"

        if landmarks:
            # Preprocess the landmarks
            processed_landmarks = preprocess_landmarks(landmarks, scaler)

            # Predict gesture
            prediction = model.predict(processed_landmarks)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            # Get gesture name
            gesture_name = gesture_classes[class_id]

            if confidence >= CONFIDENCE_THRESHOLD and gesture_name in GESTURE_ACTION_MAP:
                current_time = time.time()
                # Check if cooldown period has passed
                if current_time - last_trigger_time[gesture_name] > COOLDOWN_PERIOD:
                    # Execute the corresponding action
                    command = GESTURE_ACTION_MAP[gesture_name]
                    execute_command(command)
                    # Update the last trigger time
                    last_trigger_time[gesture_name] = current_time
                    gesture = f"{gesture_name} (Action Triggered)"
                else:
                    gesture = f"{gesture_name} (Cooldown)"
            else:
                gesture = f"{gesture_name} ({confidence*100:.2f}%)"
        else:
            gesture = "No Gesture"

        # Display the gesture on the frame
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Real-Time Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Real-Time Gesture Recognition.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()