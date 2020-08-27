import glob
import cv2 as cv
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
from ReadCameraModel import *

model_dir = r'C:\Users\sukoo\673\Project5\Oxford_dataset\model'

###################################################################################################
###################################################################################################

def K_Matrix():
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(model_dir)
    K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
    return K

K = K_Matrix()

###################################################################################################
###################################################################################################

def Homogenousmatrix(R, t):
    z = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    z = np.vstack((z, a))
    return z

###################################################################################################
###################################################################################################

def PoseEstimation_calculate(E_Matrix):
    u, s, v = np.linalg.svd(E_Matrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 1st Solution
    c1 = u[:, 2] 
    r1 = u @ w @ v
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1
    c1 = c1.reshape((3,1))
    
    # 2nd Solution
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2 
    c2 = c2.reshape((3,1))
    
    # 3rd Solution
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3 
    c3 = c3.reshape((3,1)) 
    
    # 4th Solution
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c4 = c4.reshape((3,1))
    
    return [r1, r2, r3, r4], [c1, c2, c3, c4]

###################################################################################################
###################################################################################################

def calculateEssentialMatrix(calibrationMatrix, fundMatrix):
    tempMatrix = np.matmul(np.matmul(calibrationMatrix.T, fundMatrix), calibrationMatrix)
    u, s, v = np.linalg.svd(tempMatrix, full_matrices=True)
    sigmaF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) # Constraining Eigenvalues to 1, 1, 0
    temp = np.matmul(u, sigmaF)
    E_matrix = np.matmul(temp, v)
    return E_matrix

###################################################################################################
###################################################################################################

def show(text,image):
    cv.imshow(text,image)
    cv.waitKey(1)

###################################################################################################
###################################################################################################

def gray_image(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

###################################################################################################
###################################################################################################

def keypoints_descriptors(image):
    image = cv.imread(image)
    image = gray_image(image)
    descriptor = cv.xfeatures2d.SIFT_create()
    (kpts, features) = descriptor.detectAndCompute(image, None)
    return kpts, features , image

###################################################################################################
###################################################################################################

def createMatcher():
    "Create and return a Matcher Object"
    knn = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    return knn

###################################################################################################
###################################################################################################

def fundamentalMatrix(corners1, corners2):
    A = np.empty((8, 9))

    for i in range(0, len(corners1)): # Looping over all the 8-points (features)
        x1 = corners1[i][0] # x-coordinate from current frame 
        y1 = corners1[i][1] # y-coordinate from current frame
        x2 = corners2[i][0] # x-coordinate from next frame
        y2 = corners2[i][1] # y-coordinate from next frame
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)  # Taking SVD of the matrix
    f = v[-1].reshape(3,3) # Last column of V matrix
    
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    F = u1 @ s2 @ v1  
    
    return F  

###################################################################################################
###################################################################################################

def check_value_Fmatrix(x1,x2,F): 
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

###################################################################################################
###################################################################################################

def matchKeyPointsKNN(featuresA, featuresB, ratio):
    # FLANN parameters
    # These parameters combine features from 1st image to 2nd image  
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    rawMatches = flann.knnMatch(featuresA,featuresB,k=2)
    features1 = []
    features2 = []
    matches = []
    # loop over the raw matches
    for i,(m,n) in enumerate(rawMatches):
        if m.distance < 0.5*n.distance:
            features1.append(kpsA[m.queryIdx].pt)
            features2.append(kpsB[m.trainIdx].pt)
            matches.append(m)
    return matches, features1, features2

###################################################################################################
###################################################################################################

def rotationMatrixToEulerAngles(R) :
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

###################################################################################################
###################################################################################################

def getTriangulationPoint(m1, m2, point1, point2):
    
    # Skew Symmetric Matrix of point1
    oldx = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]]) 
    # Skew Symmetric Matrix of point2
    oldxdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    
    A1 = oldx @ m1[0:3, :] 
    A2 = oldxdash @ m2
    A = np.vstack((A1, A2)) # Ax = 0
    
    u, s, v = np.linalg.svd(A)
    new1X = v[-1]
    new1X = new1X/new1X[3]
    new1X = new1X.reshape((4,1))
    
    return new1X[0:3].reshape((3,1))

###################################################################################################
###################################################################################################

def disambiguiousPose(Rlist, Clist, features1, features2):
    check = 0
    Horigin = np.identity(4) # current camera pose is always considered as an identity matrix
    for index in range(0, len(Rlist)): # Looping over all the rotation matrices
        angles = rotationMatrixToEulerAngles(Rlist[index]) # Determining the angles of the rotation matrix
        #print('angle', angles)
        
        # If the rotation of x and z axis are within the -50 to 50 degrees then it is considered down in the pipeline 
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50: 
            count = 0 
            newP = np.hstack((Rlist[index], Clist[index])) # New camera Pose 
            for i in range(0, len(features1)): # Looping over all the inliers
                temp1x = getTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i]) # Triangulating all the inliers
                thirdrow = Rlist[index][2,:].reshape((1,3)) 
                if np.squeeze(thirdrow @ (temp1x - Clist[index])) > 0: # If the depth of the triangulated point is positive
                    count = count + 1 

            if count > check: 
                check = count
                mainc = Clist[index]
                mainr = Rlist[index]
                
    if mainc[2] > 0:
        mainc = -mainc
                
    #print('mainangle', rotationMatrixToEulerAngles(mainr))
    return mainr, mainc

###################################################################################################
###################################################################################################
lastH = np.identity(4) # Initial camera Pose is considered as an identity matrix 
origin = np.array([[0, 0, 0, 1]]).T 

l = [] # Variable for storing all the trajectory points

# paths = glob.glob(r"C:\Users\Aditya Goswami\Desktop\Perception\Project 5\Oxford_dataset\Oxford_dataset\Undistorted_images\*.png")
paths = 'Undistorted_images/ '

for index in range(24, 3873):
    #Image1 gets read, Converted into gray, descriptor created and kps computed
    kpsA, featuresA, FirstImg_gray = keypoints_descriptors(paths + str(index) + '.png')
    
    #Image2 gets read, Converted into gray, descriptor created and kps computed
    kpsB, featuresB, SecImg_gray = keypoints_descriptors(paths + str(index+1) + '.png')
    
    # match keypoints calculated using matchKeyPointsKNN and matches, features1 and features2 are stored
    matches, features1, features2 = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75)
    
    # num_inlier = 0
    # final_FundMatrix = np.zeros((3,3))
    # inliers1 = []
    # inliers2 = []
    # itr = 0
    # ## RANSAC
    # while (itr<50): # 50 iterations for RANSAC 
    #     count = 0
    #     eightpoint = [] 
    #     goodFeatures1 = [] # Variable for storing eight random points from the current frame
    #     goodFeatures2 = [] # Variable for storing corresponding eight random points from the next frame
    #     tempfeature1 = [] 
    #     tempfeature2 = []

    #     while(True): # Loop runs while we do not get eight distinct random points
    #         num = random.randint(0, len(features1)-1)
    #         if num not in eightpoint:
    #             eightpoint.append(num)
    #         if len(eightpoint) == 8:
    #             break

    #     for point in eightpoint: # Looping over eight random points
    #         goodFeatures1.append([features1[point][0], features1[point][1]]) 
    #         goodFeatures2.append([features2[point][0], features2[point][1]])

    #     # Computing Fundamentals Matrix from current frame to next frame
    #     FundMatrix = fundamentalMatrix(goodFeatures1, goodFeatures2)

    #     for number in range(0, len(features1)):

    #         # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
    #         if check_value_Fmatrix(features1[number], features2[number], FundMatrix) < 0.01:
    #             count += 1 
    #             tempfeature1.append(features1[number])
    #             tempfeature2.append(features2[number])

    #     if num_inlier < count: 
    #         num_inlier = count
    #         final_FundMatrix = FundMatrix
    #         inlier1 = tempfeature1
    #         inlier2 = tempfeature2
    #     itr += 1

    # # Computing Essential Matrix from current frame to next frame        
    # E_Matrix = calculateEssentialMatrix(K, final_FundMatrix)

    # # Computing all the solutions of rotation matrix and translation vector
    # Rotation_list, Translation_list = PoseEstimation_calculate(E_Matrix)

    # # Disambiguating one solution from four
    # R, T = disambiguiousPose(Rotation_list, Translation_list, inlier1, inlier2) 

    # Estimate E, R and t using opencv built-in functions
    pts1, pts2 = np.array(features1), np.array(features2) 
    E_cv, _ = cv.findEssentialMat(pts1, pts2, focal=K[0][0], pp=(K[0][2],K[1][2]), method=cv.RANSAC, prob=0.999, threshold=0.5)
    _, R_cv, t_cv, _ = cv.recoverPose(E_cv, pts1, pts2, focal=K[0][0], pp=(K[0][2],K[1][2]))

    lastH = lastH @ Homogenousmatrix(R_cv, t_cv) # Transforming from current frame to next frame
    p = lastH @ origin # Determining the transformation of the origin from current frame to next frame

    l.append([p[0][0], -p[2][0]])
    for pts in l:
        plt.scatter(pts[0],pts[1], color = 'g')
    plt.show(block=False)
    plt.savefig(str(index)+'.png')
    plt.pause(0.1)
    plt.close()

# cv.destroyAllWindows()
# plt.show()
###################################################################################################