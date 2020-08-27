import numpy as np
import cv2 as cv

#############################################################################
def estimateForwardHomography(corners, size):
    # 4 points from src image
    xp , yp = [] , []
    for corner in corners:
        xp.append(corner[0])
        yp.append(corner[1])

    # 4 points from dst image
    width, height = size
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
    width, height = size
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
#############################################################################

def estimateHomography(corners, sizeOut=200):

    # 4 points from src image
    x = [0, sizeOut, sizeOut, 0]
    y = [0, 0, sizeOut, sizeOut]

    # 4 points from dst image
    xp , yp = [] , []
    for corner in corners:
        xp.append(corner[0])
        yp.append(corner[1])

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

    binary = preprocess(frame)
    allContours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    # eliminate any contours that don't have a parent or child
    wrongContours = [i for i, h in enumerate(hierarchy[0]) if h[2] == -1 or h[3] == -1]
    filteredContours = [c for i, c in enumerate(allContours) if i not in wrongContours]

    # retain only the 3 largest contours
    filteredContours.sort(key = cv.contourArea, reverse = True)
    filteredContours = filteredContours[:3]
    
    # extract corners based on geometry
    for contour in filteredContours:
        perimeter = cv.arcLength(contour, True)
        polygon = cv.approxPolyDP(contour, perimeter*0.11, True) # approximate the contour with a quadrilateral
        if len(polygon) == 4:
            tagContours.append(polygon)
            coordinates = [[p[0][0],p[0][1]] for p in polygon]
            tagCorners.append(coordinates)
    
    return (tagCorners, tagContours, binary)

def warp(H, img, sizeOut=200):
    warped = np.zeros((sizeOut,sizeOut), dtype=np.uint8)

    for i in range(sizeOut):
        for j in range(sizeOut):
            x, y, z = H.dot([i,j,1])
            xp, yp = int(x/z), int(y/z)
            if 0 < xp <  img.shape[1] and 0 < yp < img.shape[0]:
                warped[i][j] = img[yp][xp]

    return warped
k = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]])
k=k.T
def Projection(H, k):  # H = homographic matrix and k = camera calibration matrix
    H1 = H[:,0] # Column 1 
    H2 = H[:,1] # Column 2 
    H3 = H[:,2] # Column 3 
    # Calculating scaling factor
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(k),H1)) + np.linalg.norm(np.matmul(np.linalg.inv(k),H2)))
    B_tilda = lamda * np.matmul(np.linalg.inv(k),H)

#NOW FOR R:
    # determinant of B_tilda
    D = np.linalg.det(B_tilda) 
    if D > 0:  
        B = B_tilda #We use B_tilda as it is
    else: 
        B = B_tilda * (-1)
#R = [R1:R2:R3]
    R1 = B[:,0] 
    R2 = B[:,1] 
    R3 = np.cross(R1,R2) 
    T = B[:,2] 
    R = np.column_stack((R1, R2, R3, T)) 
    P = np.matmul(k,R)  # Multiplying calibration matrix with R matrix
    return P #P = Projection Matrix
def virtualCube(frame,P):
    #Define the 3D corner points
    
    #To calculate 2D corners : Multiplying x1,y1,z1 with Transpose of Projection Matrix
    # Calculating camera frame coordinates 
    x1,y1,z1 = np.matmul([0,0,0,1],P.T)
    x2,y2,z2 = np.matmul([0,200,0,1],P.T) 
    x3,y3,z3 = np.matmul([200,0,0,1],P.T) 
    x4,y4,z4 = np.matmul([200,200,0,1],P.T) 
    x5,y5,z5 = np.matmul([0,0,-200,1],P.T)  
    x6,y6,z6 = np.matmul([0,200,-200,1],P.T) 
    x7,y7,z7 = np.matmul([200,0,-200,1],P.T) 
    x8,y8,z8 = np.matmul([200,200,-200,1],P.T)
    
    #Drawing the Edges of Cubes
    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (0,255,0), 5) # From [0,0,0] and [0,0,200]
    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,255,0), 5) #From [0,200,0] and [0,200,200]
    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (0,255,0), 5) #From[200,0,0] and [200,0,200]
    cv.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,255,0), 5)# From [200,200,0] and [200,200,-200]
    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 5)# From [0,0,0] and [0,200,0]
    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 5)# From[0,0,0] and [200,0,0]
    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 5)# From [0,200,0] and [200,200,0]
    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 5)# From [200,0,0] and [200,200,0]
    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,255,0), 5)# From [0,0,-200] and [0,200,-200]
    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,255,0), 5)# From [0,0,-200] and [200,0,-200]
    cv.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,255,0), 5)# From [0,200,-200] and [200,200,-200]
    cv.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,255,0), 5)# From[200,0,-200] and [200,200,-200]

def decode():
        pass