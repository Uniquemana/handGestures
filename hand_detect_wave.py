import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

# Variables to track hand movement for wave detection
x_positions = []
max_positions = 20  # Number of previous positions to track (higher for smoother detection)
wave_threshold = 60  # Minimum movement in x to consider a wave (adjust for sensitivity)
wave_detected = False
cooldown_time = 0.6  # Cooldown in seconds to prevent multiple triggers
last_wave_time = 0  # To track the last time a wave was triggered

def moving_average(data, window_size=5):
    """
    Calculate the moving average of the data to smooth it out.
    """
    if len(data) < window_size:
        return data[-1]  # Return the last data point if not enough data
    return np.mean(data[-window_size:])

def detect_wave(x_positions, threshold=wave_threshold):
    """
    Detect a wave gesture based on the movement of the hand in the x-direction.
    The hand should move left-to-right and then right-to-left clearly.

    Args:
    - x_positions (list): A list of recent x-coordinates of the hand.

    Returns:
    - wave_detected (bool): True if a wave gesture is detected, False otherwise.
    """
    if len(x_positions) < max_positions:
        return False

    # Calculate the overall movement of the hand
    first_half_movement = x_positions[len(x_positions)//2] - x_positions[0]  # Left to Right
    second_half_movement = x_positions[-1] - x_positions[len(x_positions)//2]  # Right to Left

    # Detect clear wave: first move in one direction, then in the opposite
    if first_half_movement < -threshold and second_half_movement > threshold:
        return True
    if first_half_movement > threshold and second_half_movement < -threshold:
        return True

    return False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(rgb_frame)

    # Draw hand landmarks and detect wave gesture
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the wrist position (landmark 0)
            wrist = hand_landmarks.landmark[0]

            # Get the x-coordinate of the wrist relative to the frame size
            h, w, c = frame.shape
            wrist_x = int(wrist.x * w)

            # Apply moving average to smooth wrist position
            smooth_wrist_x = moving_average(x_positions + [wrist_x])

            # Add the smoothed x position to the list of previous positions
            x_positions.append(smooth_wrist_x)

            # Keep the list at a fixed size (smooth over `max_positions` frames)
            if len(x_positions) > max_positions:
                x_positions.pop(0)

            # Detect wave gesture based on smoothed x positions
            current_time = time.time()
            if detect_wave(x_positions) and not wave_detected:
                # Ensure cooldown has passed
                if current_time - last_wave_time > cooldown_time:
                    wave_detected = True
                    last_wave_time = current_time
                    cv2.putText(frame, 'Wave detected - Switching apps!', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Trigger "Cmd + Tab" with right arrow key for incremental switching
                    pyautogui.keyDown('command')  # Hold down the Command key
                    pyautogui.press('tab')  # Open the application switcher
                    pyautogui.press('right')  # Move to the next application
                    pyautogui.keyUp('command')  # Release the Command key

            # Reset wave detection after cooldown
            if current_time - last_wave_time > cooldown_time:
                wave_detected = False

    # Display the frame with landmarks
    cv2.imshow("Wave Gesture Detection - Incremental App Switching", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
