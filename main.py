import numpy as np
import cv2 as cv
from util.picture_edit import VideoEditor

cap = cv.VideoCapture("videoframe.mp4")

bg_subs = cv.createBackgroundSubtractorMOG2()
history = 300
learning_rate = 1.0 / history

arr_count_old = 0
s = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = VideoEditor.frame_edit(frame, 1)

    mask = bg_subs.apply(frame, learningRate=learning_rate)

    out = frame.copy()
    out_frame = frame.copy()
    #result = cv.bitwise_and(out, out, mask=mask)
    #cv.imshow('res', mask)

    ret, thresh1 = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)
    filter_image = VideoEditor.filter_image(thresh1, 3)
    editer = VideoEditor.mask_edit(filter_image, 3, 3)
   # cv.imshow('1', thresh1)
   # cv.imshow('2', filter_image)
   # cv.imshow('3', editer)

    arr_count = VideoEditor().find_ctr_center(editer, out)
    count = VideoEditor().car_check(arr_count)
    if arr_count_old == 1 and count == 0:
        s += 1
    arr_count_old = count
    VideoEditor().interface(out, s)
    #cv.imshow('input', filter_image)
    cv.imshow('input1', out)



    c = cv.waitKey(10)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()