#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

CAM_ID = 0
cam = cv2.VideoCapture(CAM_ID) 
if cam.isOpened() == False:
    print ('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()

#cv2.namedWindow('CAM_Window')
img_counter = 0

TEXT = "Funny Text inside the box"

while(True):
    print(img_counter)
    ret, frame = cam.read()
    if not ret:
        print(ret)
        break
    
    # (0, 255, 0) : (r,g,b)
    cv2.putText(frame, TEXT
        , (0, 100)
        , cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        , 1
        , (0, 0, 255))
    
    #cv2.imshow('CAM_Window', frame)
    cv2.imshow('img', frame)

    #Wait 100 ms
    k = cv2.waitKey(50)
    if (k % 0xFF) == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif (k % 0xFF) == ord('q'):
        # 'q'q pressed
        print("quit hit, closing...")
        break
    elif (k % 0xFF) == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
#cv2.destroyWindow('CAM_Window')


