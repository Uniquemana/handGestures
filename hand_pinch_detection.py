import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

def detect_pinch(thumb_tip, index_tip, threshold=40):
    """
    Detect if the thumb and index finger are pinched together.
    
    Args:
    - thumb_tip (tuple): (x, y) coordinates of the thumb tip (landmark 4).
    - index_tip (tuple): (x, y) coordinates of the index finger tip (landmark 8).
    - threshold (int): Maximum distance (in pixels) to consider a pinch.
    
    Returns:
    - is_pinched (bool): True if the thumb and index finger are close enough to be considered pinched.
    """
    # Calculate the Euclidean distance between the thumb tip and index finger tip
    distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
    return distance < threshold

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

    # Draw hand landmarks and detect pinch gesture
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmark positions
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Get the coordinates relative to the frame size
            h, w, c = frame.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

            # Detect pinch gesture (thumb and index finger close together)
            if detect_pinch(thumb_pos, index_tip_pos):
                cv2.putText(frame, 'Pinch detected - Closing tab!', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Trigger "Ctrl + W" to close a tab
                pyautogui.hotkey('command', 'w')

    # Display the frame with landmarks
    cv2.imshow("Gesture Detection for Tab Close", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
