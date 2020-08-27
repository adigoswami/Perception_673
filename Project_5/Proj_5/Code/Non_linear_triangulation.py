import numpy as np
from scipy.optimize import leastsq

from Project_5_Solution import getTriangulationPoint

def homog2eucl(x):
    # map from point Homogeneous to Euclidean representation
    return x[:3,:] / x[3][0]

def get_nonlinearTriangulation(pose1, pose2, point1, point2):
    estimation = getTriangulationPoint(pose1, pose2, point1, point2)
    # solve the minimization problem 
    args = (pose1, pose2, point1, point2)
    point, success = leastsq(nonlinearTriangulationError, estimation, args=args, maxfev=10000)
    
    return np.matrix(point).T

def nonlinearTriangulationError(estimation, pose1, pose2, point1, point2):
    estimation = np.array([estimation[0][0], estimation[1][0], estimation[2][0], [1]])
    estimated_point1 = homog2eucl(np.dot(pose1, estimation))
    estimated_point2 = homog2eucl(np.dot(pose2, estimation))
    estimated_point1 = np.array([estimated_point1[0][0] / estimated_point1[2][0], estimated_point1[0][0] / estimated_point1[1][0]])
    estimated_point2 = np.array([estimated_point2[0][0] / estimated_point2[2][0], estimated_point2[0][0] / estimated_point2[1][0]])
    diff1 = estimated_point1 - point1
    diff2 = estimated_point2 - point2

    return np.asarray(np.vstack([diff1,diff2]).T)[0,:]