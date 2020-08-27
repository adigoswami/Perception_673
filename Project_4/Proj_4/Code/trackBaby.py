import numpy as np
import cv2 as cv
import glob
from sys import argv

def main():
    # label = argv[1]
    label = 'Baby'
    dataset = sorted(glob.glob(f'data/{label}/img/*'))
    box_initial = {'Baby': (160,83,56,65),'Bolt': (269,75,34,64),'Car': (70,51,107,87)} # (x,y,w,h)
    p = np.zeros((6,1))
    template, initialBox, referenceFrame = acquireTemplate(dataset, box_initial[label])
    outputVideo = cv.VideoWriter(f'{label}.mp4', cv.VideoWriter_fourcc(*'XVID'), 30, (referenceFrame.shape[1], referenceFrame.shape[0]))

    for image in dataset:
        # acquire and preprocess frame
        frame = cv.imread(image)
        preprocessedFrame = preprocess(frame, referenceFrame)
        # track object
        p, boundingBox = KLT(preprocessedFrame, template, initialBox, p)
        # draw bounding box
        frame = cv.rectangle(frame, boundingBox[0], boundingBox[1], (255,0,0), 2)
        # visualize object tracking
        outputVideo.write(frame)
        imshow('Live', frame)
        if cv.waitKey(1) >= 0:
            break

def KLT(I, T, box_initial, p, iterations=50, epsilon=0.1, learningRate=100):
    tl, br = box_initial # top-left and bottom-right corners
    # pre-compute image gradient
    ksize = 5
    Ix = cv.Sobel(I, cv.CV_64F, 1, 0, ksize=ksize)
    Iy = cv.Sobel(I, cv.CV_64F, 0, 1, ksize=ksize)
    # run gradient descent
    for _ in range(iterations):
        # 1. warp image
        W = np.array([[1+p[0][0], p[2][0], p[4][0]], [p[1][0], 1+p[3][0], p[5][0]]], dtype=np.float64)
        warpedI = warpAffine(I, W, tl, br)
        # 2. compute error image
        errorImage = (T - warpedI).reshape((-1,1))
        weights = RobustLossWeights(errorImage)
        # 3. compute warped image gradient
        warpedIx = warpAffine(Ix, W, tl, br).reshape(-1,1)
        warpedIy = warpAffine(Iy, W, tl, br).reshape(-1,1)
        # 4. evaluate matrix sum (gradient * jacobian)
        SDI, i = [], 0
        for y in range(tl[1], br[1]):
            for x in range(tl[0], br[0]):
                SDI.append([x*warpedIx[i][0], x*warpedIy[i][0], y*warpedIx[i][0], y*warpedIy[i][0], warpedIx[i][0], warpedIy[i][0]])
                if errorImage[i][0] < -50 or errorImage[i][0] > 50:
                    errorImage[i][0] = 0
                i += 1
        SDI = np.array(SDI)
        matrixSum = SDI.T.dot(errorImage*weights)
        # 5. compute hessian
        H = SDI.T.dot(SDI)
        Hinv = np.linalg.pinv(H)
        # 6. compute delta p
        delta_p = Hinv.dot(matrixSum)
        # 7. update parameters
        p += learningRate * delta_p
        # 8. check for convergence
        if (np.linalg.norm(delta_p) < epsilon):
            break
    # predict new bounding box
    W = np.array([[1+p[0][0], p[2][0], p[4][0]], [p[1][0], 1+p[3][0], p[5][0]]], dtype=np.float64)
    tl, br = np.array([tl[0], tl[1], 1]), np.array([br[0], br[1], 1])
    tl_new, br_new = W.dot(tl).astype(int), W.dot(br).astype(int)
    box_new = (((max(tl_new[0],br_new[0])-50), tl_new[1]), (br_new[0], br_new[1]))

    return (p, box_new)

def RobustLossWeights(error):
    var = np.var(error)
    sd = np.sqrt(var)
    mean = np.mean(error)
    n,it = error.shape
    W = np.zeros((n,it))
    w1,w2 = np.where(np.abs(mean - error) <= var)
    W[w1,w2] = 0.4
    w3,w4 = np.where(np.abs(mean - error) > var)
    W[w3,w4] = 0.1
    # W = [w1,w2,w3,w4]
    return W

def acquireTemplate(dataset, box_initial):
    x, y, w, h = box_initial
    frame = cv.imread(dataset[0])
    referenceFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    referenceFrame = cv.GaussianBlur(referenceFrame, (5, 5), 5)  
    templateFrame = referenceFrame[y:y+h,x:x+w]
    templateBox = ((x,y),(x+w,y+h)) # top left and bottom right corners

    return (templateFrame, templateBox, referenceFrame)

def preprocess(frame, template):
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayFrame = cv.GaussianBlur(grayFrame, (5, 5), 5)  
    
    return grayFrame * np.mean(template) / np.mean(grayFrame) # enforce brightness constancy across frames

def warpAffine(image, affineTransformation, topLeft, bottomRight):
    warped = cv.warpAffine(image, affineTransformation, (0,0), flags=cv.WARP_INVERSE_MAP) 
    
    return warped[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

def imshow(windowName, image):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, image)

if __name__ == '__main__':
    main()