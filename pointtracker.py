import cv2
import numpy as np

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture(1)
video_1 = cv2.VideoCapture(0)

ok,frame=video.read()

bbox = cv2.selectROI(frame)

ok = tracker.init(frame,bbox)

# print 
cx=0
cy=0

while True:
    print(cx,cy)
    ok,frame=video.read()
    ok_, frame2 = video_1.read()
    if not ok:
        break
    ok,bbox=tracker.update(frame)
    ok_,bbox2=tracker.update(frame2)
    if ok:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2,1)
        
        cx,cy = x+w/2,y+h/2
        
    else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Tracking',frame)
    cv2.imshow('Tracking2',frame2)
    
    if cv2.waitKey(1) & 0XFF==27:
        break
cv2.destroyAllWindows()