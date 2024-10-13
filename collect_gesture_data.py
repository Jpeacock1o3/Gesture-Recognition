import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

# ---------------------------- Configuration ----------------------------

# Define your custom gestures here
GESTURES = ['Wave', 'Point', 'Heart', 'Thumbs Up', 'Okay']  # Add more gestures as needed

# Directory to save the collected gesture data
DATA_DIR = 'gesture_data'  # You can change this to your preferred directory
os.makedirs(DATA_DIR, exist_ok=True)

# Number of samples to collect per gesture
NUM_SAMPLES_PER_GESTURE = 200  # Adjust based on your needs

# MediaPipe Hands Configuration
MAX_NUM_HANDS = 1  # Number of hands to detect
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# --------------------------- Initialize MediaPipe ---------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # Video stream
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

# ----------------------------- Helper Functions -----------------------------

def collect_data_for_gesture(gesture_name, num_samples=NUM_SAMPLES_PER_GESTURE):
    """
    Captures gesture data from the webcam and saves it as JSON files.

    Args:
        gesture_name (str): The name of the gesture to collect.
        num_samples (int): Number of samples to collect for the gesture.
    """
    cap = cv2.VideoCapture(0)  # Initialize webcam (0 is the default camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    collected = 0
    print(f"Collecting data for gesture: '{gesture_name}'")
    print("Press 'q' to quit early.")

    while collected < num_samples:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(img_rgb)

        # Draw landmarks and connections if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Create a dictionary for the gesture
                gesture_data = {
                    'landmarks': landmarks,
                    'label': gesture_name
                }

                # Define the file path
                timestamp = int(time.time() * 1000)
                filename = f"{gesture_name}_{timestamp}.json"
                file_path = os.path.join(DATA_DIR, filename)

                # Save the gesture data as JSON
                with open(file_path, 'w') as f:
                    json.dump(gesture_data, f)

                collected += 1
                print(f"Collected {collected}/{num_samples} for gesture '{gesture_name}'")

                # Break after the first hand is processed
                break

        # Display the frame
        cv2.imshow(f"Collecting Gesture: {gesture_name}", frame)

        # Exit early if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Data collection interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection for gesture '{gesture_name}' completed.")

def main():
    """
    Main function to collect data for all defined gestures.
    """
    for gesture in GESTURES:
        collect_data_for_gesture(gesture_name=gesture, num_samples=NUM_SAMPLES_PER_GESTURE)
        # Optional: Add a short pause between gestures
        time.sleep(2)

if __name__ == "__main__":
    main()