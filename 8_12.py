#!/bin/env python3.8
import cv2
import mediapipe as mp
import copy
import math

import numpy
numpy.random.BitGenerator = numpy.random.bit_generator.BitGenerator

import pandas as pd

mp_hands = mp.solutions.hands

columns__=['Good_FrameNumber']
for i in range(21):
    for j in ['x', 'y', 'z']:
        columns__.append(str(i)+'_'+j)

df = pd.DataFrame(columns = columns__)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("Fist.mp4")

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
ctr=0

S0 = {}
S0_prime = {}

Angles={
    'TI':-999,
    'IM':-999,
    'MR':-999,
    'RP':-999
}

def AnglesCalc__(State):
    v1 = (State['THUMB_TIP'][0]-State['W2T'][0], State['THUMB_TIP'][1]-State['W2T'][1])
    v2 = (State['WRIST'][0]-State['W2T'][0], State['WRIST'][1]-State['W2T'][1])
    
    Angles['WT'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

    v1 = (State['MIDDLE_FINGER_TIP'][0]-State['W2M'][0], State['MIDDLE_FINGER_TIP'][1]-State['W2M'][1])
    v2 = (State['WRIST'][0]-State['W2M'][0], State['WRIST'][1]-State['W2M'][1])
    
    Angles['WM'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

    v1 = (State['RING_FINGER_TIP'][0]-State['W2R'][0], State['RING_FINGER_TIP'][1]-State['W2R'][1])
    v2 = (State['WRIST'][0]-State['W2R'][0], State['WRIST'][1]-State['W2R'][1])
    
    Angles['WR'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

    v1 = (State['INDEX_FINGER_TIP'][0]-State['W2I'][0], State['INDEX_FINGER_TIP'][1]-State['W2I'][1])
    v2 = (State['WRIST'][0]-State['W2I'][0], State['WRIST'][1]-State['W2I'][1])
    
    Angles['WI'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))



with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        
        while(True):
            success, img = cap.read()
            if(success==False):
                continue

            imgRGB = cv2.flip(img,1)
            results = hands.process(imgRGB)

            cv2.imshow("Image", imgRGB)
            signal = cv2.waitKey(1)
            if(signal==99):
                break
        
        y__ =1
        img_height, img_width, _ = img.shape

        print('SHAPE : ', img_height, img_width, _)
        print(results)


        # Assumption : Only 1 hand under observation
        try:
            coordinates_info = results.multi_hand_landmarks[0]
        except:
            continue

        if(True):
            S0_prime['WRIST'] = (coordinates_info.landmark[mp_hands.HandLandmark.WRIST].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.WRIST].y * img_height)
            S0_prime['W2I'] = (coordinates_info.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * img_height)
            S0_prime['INDEX_FINGER_TIP'] = (coordinates_info.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img_height)
            S0_prime['W2T'] = (coordinates_info.landmark[mp_hands.HandLandmark.THUMB_CMC].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.THUMB_CMC].y * img_height)
            S0_prime['THUMB_TIP'] = (coordinates_info.landmark[mp_hands.HandLandmark.THUMB_TIP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.THUMB_TIP].y * img_height)
            S0_prime['W2M'] = (coordinates_info.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * img_height)
            S0_prime['MIDDLE_FINGER_TIP'] = (coordinates_info.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * img_height)
            S0_prime['W2R'] = (coordinates_info.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * img_height)
            S0_prime['RING_FINGER_TIP'] = (coordinates_info.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * img_height)
            S0_prime['W2P'] = (coordinates_info.landmark[mp_hands.HandLandmark.PINKY_MCP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.PINKY_MCP].y * img_height)
            S0_prime['PINKY_TIP'] = (coordinates_info.landmark[mp_hands.HandLandmark.PINKY_TIP].x * img_width, coordinates_info.landmark[mp_hands.HandLandmark.PINKY_TIP].y * img_height)
            file = open('S0_Prime.txt', 'wt')
            file.write(str(S0_prime))
            file.close()
            # break

        def StateChange(Original, Terget):
            Original=copy.deepcopy(Terget)
            return Original

        def AnglesCalc(State):
            v1 = (State['THUMB_TIP'][0]-State['WRIST'][0], State['THUMB_TIP'][1]-State['WRIST'][1])
            v2 = (State['INDEX_FINGER_TIP'][0]-State['WRIST'][0], State['INDEX_FINGER_TIP'][1]-State['WRIST'][1])
            
            Angles['TI'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

            v1 = (State['MIDDLE_FINGER_TIP'][0]-State['WRIST'][0], State['MIDDLE_FINGER_TIP'][1]-State['WRIST'][1])
            v2 = (State['INDEX_FINGER_TIP'][0]-State['WRIST'][0], State['INDEX_FINGER_TIP'][1]-State['WRIST'][1])
            
            Angles['IM'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

            v1 = (State['MIDDLE_FINGER_TIP'][0]-State['WRIST'][0], State['MIDDLE_FINGER_TIP'][1]-State['WRIST'][1])
            v2 = (State['RING_FINGER_TIP'][0]-State['WRIST'][0], State['RING_FINGER_TIP'][1]-State['WRIST'][1])
            
            Angles['MR'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

            v1 = (State['RING_FINGER_TIP'][0]-State['WRIST'][0], State['RING_FINGER_TIP'][1]-State['WRIST'][1])
            v2 = (State['PINKY_TIP'][0]-State['WRIST'][0], State['PINKY_TIP'][1]-State['WRIST'][1])
            
            Angles['RP'] = math.acos((v1[0]*v2[0]+v1[1]*v2[1])/(0.0001+((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2))**0.5))

        AnglesCalc(S0_prime)
        print('Angles = ', Angles)
        Angles = {}
        AnglesCalc__(S0_prime)
        print('Angles_ = ', Angles)

        

        print('\nBefoe: ')
        print('S0 = ', S0, '\nS0_prime = ', S0_prime)
        S0 = StateChange(S0, S0_prime)
        print('\nS0 = ', S0, '\nS0_prime = ', S0_prime)

