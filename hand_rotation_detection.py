import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

def detect_hand_lean(wrist_pos, index_tip_pos):
    """
    Detect the lean of the hand (left or right) based on the x-coordinates
    of the wrist and index finger tip.
    
    Args:
    - wrist_pos (tuple): (x, y) coordinates of the wrist (landmark 0).
    - index_tip_pos (tuple): (x, y) coordinates of the index finger tip (landmark 8).
    
    Returns:
    - lean_direction (str): "Clockwise" if leaning right, "Anti-clockwise" if leaning left, "Neutral" if straight.
    """
    wrist_x = wrist_pos[0]
    index_x = index_tip_pos[0]
    
    # Determine if the index finger is to the right or left of the wrist
    if index_x > wrist_x + 50:  # Adjust threshold for stability
        return "Clockwise"  # Hand leaning right
    elif index_x < wrist_x - 50:  # Adjust threshold for stability
        return "Anti-clockwise"  # Hand leaning left
    else:
        return "Neutral"  # Hand is straight

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

    # Draw hand landmarks and detect lean direction
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmark positions
            wrist = hand_landmarks.landmark[0]
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Get the coordinates relative to the frame size
            h, w, c = frame.shape
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            index_tip_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

            # Detect hand lean (clockwise or anti-clockwise)
            lean_direction = detect_hand_lean(wrist_pos, index_tip_pos)

            # Display the lean direction on the frame
            cv2.putText(frame, f'Lean: {lean_direction}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with landmarks
    cv2.imshow("Hand Lean Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
