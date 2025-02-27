import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[:2]

# Open webcam
cap = cv2.VideoCapture(0)

# Store last volume level
last_volume_level = 0
crossed_fingers_prev = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    hand_count = 0
    total_fingers = 0
    crossed_fingers = False

    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = hand_landmarks.landmark
            finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb to pinky

            # Count extended fingers
            fingers = [1 if tip.y < landmarks[i - 2].y else 0 for i, tip in zip([8, 12, 16, 20], finger_tips[1:])]
            fingers.insert(0, 1 if abs(finger_tips[0].x - landmarks[2].x) > 0.05 else 0)  # Thumb condition

            total_fingers += sum(fingers)

        # Adjust volume based on hand count
        if hand_count == 1:
            volume_level = np.interp(total_fingers, [0, 5], [min_vol, max_vol])
        else:
            volume_level = np.interp(total_fingers, [0, 10], [min_vol, max_vol])

        last_volume_level = volume_level
        volume.SetMasterVolumeLevel(volume_level, None)

        # Check for crossed fingers (Index and Middle)
        if len(result.multi_hand_landmarks) > 1:  # Ensure two hands are detected
            hand1 = result.multi_hand_landmarks[0].landmark
            hand2 = result.multi_hand_landmarks[1].landmark

            index1, middle1 = hand1[8], hand1[12]
            index2, middle2 = hand2[8], hand2[12]

            if abs(index1.x - index2.x) < 0.05 and abs(middle1.x - middle2.x) < 0.05:
                crossed_fingers = True

 

    crossed_fingers_prev = crossed_fingers

    # Display text info
    cv2.putText(frame, f'Volume: {total_fingers}/{10 if hand_count > 1 else 5}', 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Detection & Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
