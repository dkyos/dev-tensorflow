#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

CAM_ID = 0

cam = cv2.VideoCapture(CAM_ID) #카메라 생성
if cam.isOpened() == False: #카메라 생성 확인
    print ('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()

#윈도우 생성 및 사이즈 변경
cv2.namedWindow('CAM_Window')

########### 추가 ##################
# 문자열 저장
str = "Funny Text inside the box"
###################################
while(True):
    #카메라에서 이미지 얻기
    ret, frame = cam.read()
    
    ########### 추가 ##################
    # frame이라는 이미지에 글씨 넣는 함수
    # frame : 카메라 이미지
    # str : 문자열 변수
    # (0, 100) : 문자열이 표시될 좌표 x = 0, y = 100
    # cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 폰트 형태
    # 1 : 문자열 크기(scale) 소수점 사용가능
    # (0, 255, 0) : 문자열 색상 (r,g,b)
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255))
    ###################################
    
    #얻어온 이미지 윈도우에 표시
    cv2.imshow('CAM_Window', frame)

    #10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
       break;

#윈도우 종려
cam.release()
cv2.destroyWindow('CAM_Window')
