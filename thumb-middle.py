import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
print(f"Default FPS: {cap.get(cv2.CAP_PROP_FPS)}")
cap.set(cv2.CAP_PROP_FPS, 120)  # Try setting to 120 FPS
print(f"Set FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Helper function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Initialize variables
click_state = False
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to pass by reference
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                thumb_index_distance = calculate_distance(thumb_tip, index_tip)
                index_middle_distance = calculate_distance(index_tip, middle_tip)

                # Check for click gesture (thumb very close to index finger)
                if thumb_index_distance < 0.035 and index_middle_distance > 0.025:
                    if not click_state:
                        pyautogui.click()
                        click_state = True
                else:
                    click_state = False

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display distances for debugging
                cv2.putText(image, f"T-I: {thumb_index_distance:.3f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"I-M: {index_middle_distance:.3f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display FPS
    cv2.putText(image, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Left Hand Detection and Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()