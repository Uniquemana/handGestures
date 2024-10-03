import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen size for mouse control
screen_width, screen_height = pyautogui.size()

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

def move_mouse_based_on_index_finger(index_finger_x, index_finger_y, frame_width, frame_height):
    """
    Move the mouse based on the normalized position of the index finger.
    
    Args:
    - index_finger_x (float): x-coordinate of the index finger (normalized).
    - index_finger_y (float): y-coordinate of the index finger (normalized).
    - frame_width (int): width of the camera frame.
    - frame_height (int): height of the camera frame.
    """
    # Convert the index finger coordinates to screen coordinates
    screen_x = int(index_finger_x * screen_width)
    screen_y = int(index_finger_y * screen_height)

    # Move the mouse to the new position
    pyautogui.moveTo(screen_x, screen_y)


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

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and display the coordinates of the fingertips
            index_finger_id = 8  # Index finger tip
            h, w, c = frame.shape  # Get the frame dimensions
            
            # Get the coordinates of the index finger tip
            index_finger_x = hand_landmarks.landmark[index_finger_id].x
            index_finger_y = hand_landmarks.landmark[index_finger_id].y

            # Draw a circle on the index finger tip
            cx = int(index_finger_x * w)
            cy = int(index_finger_y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, 'Index Finger', (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Call the function to move the mouse
            move_mouse_based_on_index_finger(index_finger_x, index_finger_y, w, h)

    # Display the frame with landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
