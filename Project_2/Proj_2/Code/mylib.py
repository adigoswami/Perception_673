import numpy as np
import cv2 as cv

leftFit_prev, rightFit_prev = [], []

def imshow(windowName, image):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, image)

def laneFit(binaryFrame, polyDegree):
    global leftFit_prev
    global rightFit_prev

    #Hyper-parameters
    no_windows = 9 # number of sliding windows
    windowWidth = 100 # +/- margin
    minPixels = 50 # minimum number of pixels found to recenter window
    
    processedFrame = np.dstack((binaryFrame, binaryFrame, binaryFrame))

    histogram = np.sum(binaryFrame[binaryFrame.shape[0]//2:,:], axis=0) # only for the bottom half of the image
    windowHeight = np.int(binaryFrame.shape[0]/no_windows)

    midpoint = np.int(histogram.size/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = binaryFrame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftLanePixels, rightLanePixels = [], []

    for window in range(no_windows):
        # Define the windows
        winy_low = binaryFrame.shape[0] - (window+1)*windowHeight
        winy_high = binaryFrame.shape[0] - window*windowHeight
        leftwinx_low = leftx_base - windowWidth
        leftwinx_high = leftx_base + windowWidth
        rightwinx_low = rightx_base - windowWidth
        rightwinx_high = rightx_base + windowWidth
        # Draw the windows
        cv.rectangle(processedFrame,(leftwinx_low,winy_low),(leftwinx_high,winy_high),(0,255,0), 2)
        cv.rectangle(processedFrame,(rightwinx_low,winy_low),(rightwinx_high,winy_high),(0,255,0), 2)
        # Find nonzero pixels in x and y inside the windows
        leftPixels = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= leftwinx_low) & (nonzerox < leftwinx_high)).nonzero()[0]
        rightPixels = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= rightwinx_low) & (nonzerox < rightwinx_high)).nonzero()[0]
        leftLanePixels.append(leftPixels)
        rightLanePixels.append(rightPixels)
        # Recenter next windows if necessary
        if len(leftPixels) > minPixels:
            leftx_base = np.int(np.mean(nonzerox[leftPixels]))
        if len(rightPixels) > minPixels:
            rightx_base = np.int(np.mean(nonzerox[rightPixels]))

    leftLanePixels = np.concatenate(leftLanePixels)
    rightLanePixels = np.concatenate(rightLanePixels)

    leftx = nonzerox[leftLanePixels]
    lefty = nonzeroy[leftLanePixels]
    rightx = nonzerox[rightLanePixels]
    righty = nonzeroy[rightLanePixels]

    if lefty.size == 0:
        leftFit = leftFit_prev
    else:
        leftFit = np.polyfit(lefty, leftx, polyDegree)
        leftFit_prev = leftFit

    if righty.size == 0:
        rightFit = rightFit_prev
    else:
        rightFit = np.polyfit(righty, rightx, polyDegree)
        rightFit_prev = rightFit
    
    processedFrame[nonzeroy[leftLanePixels], nonzerox[leftLanePixels]] = [255, 0, 0]
    processedFrame[nonzeroy[rightLanePixels], nonzerox[rightLanePixels]] = [0, 0, 255]

    return leftFit, rightFit, processedFrame

def binarize(i, frame):
    # Camera Matrix (K) & Distortion Coefficients (dist)
    if i == 0: 
        K = np.array([
            [1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
            [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ], dtype=np.float32)
        dist = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02], dtype=np.float32)

        undistorted = cv.undistort(frame, K, dist)    
        warped = warp(i, undistorted, undo=False)
       
        yellow_lower = np.array([0,120,51])
        yellow_upper = np.array([33,157,255])
        white_lower = np.array([0,0,205])
        white_upper = np.array([255,255,255])

        hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV) 
        yellow_mask = cv.inRange(hls, yellow_lower, yellow_upper)
        white_mask = cv.inRange(hsv, white_lower, white_upper)
        binaryFrame = cv.bitwise_or(yellow_mask, white_mask)

    else:
        K = np.array([
            [9.037596e+02, 0.000000e+00, 6.957519e+02], 
            [0.000000e+00, 9.019653e+02, 2.242509e+02], 
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ], dtype = np.float32)
        dist = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]], dtype = np.float32)

        undistorted = cv.undistort(frame, K, dist)    
        warped = warp(i, undistorted, undo=False)

        hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        whiteLower = np.array([0, 0, 211]) 
        whiteUpper = np.array([255, 11, 255])
        white_mask = cv.inRange(hls, whiteLower, whiteUpper)

        frame = cv.bitwise_or(gray, white_mask)
        thresh = 159
        binaryFrame = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)[1]

    return binaryFrame

def constructRegressionMatrix(x_dataset, degree):
    X_dataset = np.ones((len(x_dataset),1))
    for i in range(1,degree+1):
        X_dataset = np.column_stack((x_dataset**i, X_dataset))

    return X_dataset

def warp(i, frame, undo):
    if  i==0: 
        src = np.array([
            [190, 700], 
            [1110, 700], 
            [720, 470], 
            [570, 470]
        ], dtype=np.float32)
        dst = np.array([
            [250, 720],
            [1035, 720],
            [1035, 0],
            [250, 0],

        ], dtype=np.float32)

    else: 
        src = np.array([
            [203, 510],
            [545, 300],
            [745, 300],
            [940, 510]
        ], dtype=np.float32)
        dst = np.array([
             [250, 510],
             [250, 0],
             [1140, 0],
             [1140, 510]
         ], dtype=np.float32)
        
    if undo:
        src, dst = dst, src

    frameSize = (frame.shape[1], frame.shape[0])
    H = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(frame, H, frameSize) # keep same size as input image

    return warped