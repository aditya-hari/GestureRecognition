import cv2
import numpy as np
from predictor import predictor
from tkinter import *

def cropped(frame):
    crop = frame[100:350, 400:650]
    return crop

def capture():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(frame,100,200)
        roi = cropped(edges)

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break
        if keypress ==  ord('s'):
            save(roi)
            value = predictor()
            print(value)
        cv2.imshow("ROI", roi)
    cap.release()
    cv2.destroyAllWindows()

def save(img):
    img = cv2.Canny(img,100,200)

    kernel1 = np.ones((3,3),np.uint8)

    img = cv2.dilate(img,kernel1,iterations = 2)
    img = cv2.erode(img,kernel1,iterations = 1)

    im_floodfill = img.copy()

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    fill = cv2.bitwise_not(im_floodfill)
    ret,thresh = cv2.threshold(fill,25,255,cv2.THRESH_BINARY)
    fill = cv2.resize(fill, (64,64))
    cv2.imwrite("cap.jpg",fill)

capture()