import cv2 as cv
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

cap = cv.VideoCapture('data/night.mp4')
out = cv.VideoWriter('output/p1.mp4',cv.VideoWriter_fourcc(*'XVID'), 30, (1920,1080))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    #if (cap.isOpened()):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
 
    adjusted = adjust_gamma(frame, gamma=3.0)
    out.write(adjusted)
    #show the two videos as output (side by side)
    cv.imshow('Frame', adjusted)
    #press escape to exit
    if cv.waitKey(1) == 27:
        break
    
cap.release()
out.release()
cv.destroyAllWindows()