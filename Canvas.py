import cv2
import mediapipe as mp
import numpy as np

# Initialize hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Get webcam dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Drawing parameters
canvas = np.zeros((height, width, 3), dtype=np.uint8)
drawing_mode = "draw"
last_x, last_y = None, None
eraser_thickness = 50
drawing_thickness = 5

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    results = hands.process(image)

    # Draw landmarks and connections if found
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip coordinates
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Gesture recognition
            if index_tip.y < middle_tip.y and index_tip.y < ring_tip.y:
                drawing_mode = "draw"
            elif index_tip.y > middle_tip.y and index_tip.y > ring_tip.y:
                drawing_mode = "move"
            elif middle_tip.y < index_tip.y and middle_tip.y < ring_tip.y:
                drawing_mode = "erase"

            # Drawing logic
            x, y = int(index_tip.x * width), int(index_tip.y * height)
            if drawing_mode == "draw":
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (x, y), (0, 255, 0), drawing_thickness)
                last_x, last_y = x, y
            elif drawing_mode == "erase":
                cv2.circle(canvas, (x, y), eraser_thickness, (0, 0, 0), -1)
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (x, y), (0, 0, 0), eraser_thickness * 2)
                last_x, last_y = x, y
            elif drawing_mode == "move":
                last_x, last_y = None, None

    # Display camera feed and canvas side-by-side
    combined = np.hstack([image, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)])
    cv2.namedWindow('Virtual Canvas', cv2.WINDOW_NORMAL)
    cv2.imshow('Virtual Canvas', combined)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()