import numpy as np
import cv2 as cv
from scipy import stats

def imshow(windowName, image):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, image)

def estimateForwardHomography(corners, size):
    # 4 points from src image
    xp , yp = [] , []
    for corner in corners:
        xp.append(corner[0])
        yp.append(corner[1])

    # 4 points from dst image
    height, width = size
    x = [0, width, width, 0]
    y = [0, 0, height, height]

    A = np.array([
        [-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
        [0,0,0,-x[0],-y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
        [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
        [0,0,0,-x[1],-y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
        [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
        [0,0,0,-x[2],-y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
        [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
        [0,0,0,-x[3],-y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]],
    ], dtype=np.float64)

    *_,V = np.linalg.svd(A)
    return np.reshape(V[-1,:],(3,3))

def estimateInverseHomography(corners, size):
    # 4 points from src image
    x , y = [] , []
    for corner in corners:
        x.append(corner[0])
        y.append(corner[1])

    # 4 points from dst image
    height, width = size
    xp = [0, width, width, 0]
    yp = [0, 0, height, height]

    A = np.array([
        [-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
        [0,0,0,-x[0],-y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
        [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
        [0,0,0,-x[1],-y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
        [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
        [0,0,0,-x[2],-y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
        [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
        [0,0,0,-x[3],-y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]],
    ], dtype=np.float64)

    *_,V = np.linalg.svd(A)
    return np.reshape(V[-1,:],(3,3))

def preprocess(frame):
    #frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscale,(5,5),0) # try median blur as well
    _, binary = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)

    return binary

def findCorners(frame):
    tagContours, tagCorners = [], []

    binaryFrame = preprocess(frame)
    allContours, hierarchy = cv.findContours(binaryFrame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    # eliminate any contours that don't have a parent or child
    wrongContours = [i for i, h in enumerate(hierarchy[0]) if h[2] == -1 or h[3] == -1]
    filteredContours = [c for i, c in enumerate(allContours) if i not in wrongContours]

    # retain only the 3 largest contours
    filteredContours.sort(key = cv.contourArea, reverse = True)
    filteredContours = filteredContours[:3]
    
    # extract corners based on geometry
    for contour in filteredContours:
        perimeter = cv.arcLength(contour, True)
        polygon = cv.approxPolyDP(contour, perimeter*0.015, True) # approximate the contour with a quadrilateral
        if len(polygon) == 4:
            tagContours.append(polygon)
            coordinates = [[p[0][0],p[0][1]] for p in polygon]
            tagCorners.append(coordinates)
    
    return (tagCorners, tagContours, binaryFrame)

def warp(H, frame, size):
    
    height, width = size[0:2]

    if len(size) == 3:
        warped = np.zeros((height,width,size[2]), dtype=np.uint8)
    else:
        warped = np.zeros((height,width), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            xout, yout, zout = H.dot([x,y,1])
            xp, yp = int(xout/zout), int(yout/zout)
            if 0 < xp <  frame.shape[1] and 0 < yp < frame.shape[0]:
                if len(size) == 3:
                    warped[y,x,:] = frame[yp,xp,:]
                else:
                    warped[y][x] = frame[yp][xp]

    return warped

def rotate(frame, orientation):
    if orientation == 90:
        return cv.rotate(frame,cv.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 180:
        return cv.rotate(frame,cv.ROTATE_180)
    elif orientation == -90:
        return cv.rotate(frame,cv.ROTATE_90_CLOCKWISE)
    else:
        return frame
        
def decode(binaryTag):
    binaryKey = key.generate(binaryTag)

    b0, b1, b2, b3 = str(binaryKey[2][2]), str(binaryKey[1][2]), str(binaryKey[1][1]), str(binaryKey[2][1])

    # 0 -90 180 90
    if binaryKey[3,3] == 1:
        orientation = 0 
        ID = b0+b1+b2+b3
    elif binaryKey[0,3] == 1:
        orientation = +90
        ID = b3+b0+b1+b2
    elif binaryKey[0,0] == 1:
        orientation = 180
        ID = b2+b3+b0+b1
    elif binaryKey[3,0] == 1:
        orientation = -90
        ID = b1+b2+b3+b0
    else:
        orientation = 0
        ID = '0000'
        
    return (orientation, ID)

def renderMask(frame,contour):
    cv.drawContours(frame,[contour],-1,(0,0,0),thickness=-1)

    return frame

class key:
    prev = np.zeros((4,4), dtype=np.uint8)

    @classmethod
    def generate(cls, binaryTag):
        tagSize = binaryTag.shape[0]
        gridCellSize = 8
        step = tagSize // gridCellSize
        binaryKey = np.zeros((4,4), dtype=np.uint8)

        row, col = 0, 0
        for y in range(2*step, tagSize-2*step, step):
            for x in range(2*step, tagSize-2*step, step):
                roi = binaryTag[y:y+step, x:x+step]
                if stats.mode(roi, axis=None)[0] == 255:
                    binaryKey[row][col] = 1
                col += 1
            row += 1
            col = 0

        if binaryKey[0][0] + binaryKey[-1][-1] + binaryKey[0][-1] + binaryKey[-1][0] == 1:
            key.prev = binaryKey
            return binaryKey
        else:
            return key.prev