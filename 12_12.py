import numpy as np
import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=0,
                      max_num_hands=1,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils


video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

cx_0,cy_0,cx_1,cy_1=0,0,0,0
while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    imgRGB = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame0.shape
                cx_0, cy_0 = int(lm.x *w), int(lm.y*h)
                # cv2.circle(frame0, (cx_0,cy_0), 5, (255,255,255), cv2.FILLED)
                # if(id>0):
                break
            # mpDraw.draw_landmarks(frame0, handLms, mpHands.HAND_CONNECTIONS)

    imgRGB = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame1.shape
                cx_1, cy_1 = int(lm.x *w), int(lm.y*h)
                # cv2.circle(frame1, (cx_1,cy_1), 5, (255,255,255), cv2.FILLED)
                # if(id>0):
                break
            # mpDraw.draw_landmarks(frame1, handLms, mpHands.HAND_CONNECTIONS)

    # (coordinates_info.landmark[mp_hands.HandLandmark.WRIST].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.WRIST].y * img_height)
    if(cx_0&cx_1&cy_0&cy_1):
        print('Coordinates from camera 0 = ({}, {})'.format(cx_0, cy_0))
        print('Coordinates from camera 1 = ({}, {})'.format(cx_1, cy_1))

        c1 = np.array([cx_0, cy_0])
        c2 = np.array([cx_1, cy_1])

        print('L1 norm = {} \tL2 norm = {}\n\n'.format(np.linalg.norm(c1-c2,1), np.linalg.norm(c1-c2,2)))

    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', cv2.flip(frame0,1))

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', cv2.flip(frame1,1))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
