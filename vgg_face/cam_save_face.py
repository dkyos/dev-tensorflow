#!/usr/bin/env python

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from tkinter import *
import tkinter as tk
import tkinter.simpledialog as tkSimpleDialog

class MyDialog(tkSimpleDialog.Dialog):
    def body(self, master):
        self.geometry("400x200")
        tk.Label(master, text="Save name :").grid(row=0)

        self.e1 = tk.Entry(master)
        self.e1.grid(row=0, column=1)
        self.e1.config(bg='black', fg='yellow')

        labelfont = ('times', 20, 'bold')
        self.e1.config(font=labelfont)

        return self.e1 # initial focus

    def apply(self):
        first = self.e1.get()
        self.result = first

root = tk.Tk()
root.withdraw()

#color = (67,67,67)
color = (200,100,200)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#webcam
CAM_ID = 0
cap = cv2.VideoCapture(CAM_ID) 
if cap.isOpened() == False:
    print ('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()

iteration = 0

criterion = 80

while(True):
    x = y = w = h = 0
    if iteration % 10 == 0:
        print("Detecting Face : " + str(iteration))
    iteration += 1
    ret, img = cap.read()
    if not ret:
        print(ret)
        break

    #img = cv2.resize(img, (640, 360))
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in faces:
        if w > criterion: 
            #draw rectangle to main image
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            print("Face with (%d %d %d %d)" % (x, y, w, h))
            
            #crop detected face without lines 
            detected_face = img[int(y+2):int(y+h-2)
                , int(x+2):int(x+w-2)] 
            #resize to 224x224
            detected_face = cv2.resize(detected_face, (224, 224))
            
            #connect face and text
            cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),color,1)
            cv2.line(img,(x+w,y-20),(x+w+10,y-20),color,1)
        else:
            print("No Face with (%d %d %d %d" % (x, y, w, h))
        
    cv2.imshow('img',img)
    
    k = cv2.waitKey(10)
    if (k % 0xFF) == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif (k % 0xFF) == ord('q'):
        # 'q'q pressed
        print("quit hit, closing...")
        break
    elif ((k % 0xFF) == ord('s')) | ((k % 0xFF) == 32):
        # SPACE pressed
        if w > criterion: 
            text = MyDialog(root, "Saving image")
            print (text.result)
            name = text.result
            img_name = "./database/{}.png".format(name)
            #cv2.imwrite(img_name, img)
            cv2.imwrite(img_name, detected_face)
            print("{} written!".format(img_name))
        else:
            print ("No face")
    
#kill open cv things        
cap.release()
cv2.destroyAllWindows()




