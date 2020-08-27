
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import csv


#To check whether a point lies within the threshold
#Thus finding the inliers and outliers
def check_whether_outlier_inlier(a,b,c,xi,yi,th):
   
    dist = yi -(a*(xi**2)) - (b*xi) - c
   
    if abs(dist) <= th:             # if the absolute distance <= threshold -> It is an inlier
        return 1
    else:
        return 0

#To check if the points which are randomly picked aren't repeatedly picked
def Repeating_or_not(PointsList, PointsSet):
	check = 0
	for p in PointsSet:
		if (PointsList[0] in p) and (PointsList[1] in p) and (PointsList[2] in p):
			check = 1
	return check

if __name__== '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', "--input_file", required=False, help="Input csv file containing data",
            default='data_1.csv', type=str)

    args = vars(ap.parse_args())
    
    threshold = 25  
    N = 4000        # Number of samples or iterations
    x=[]
    y=[]
    with open(args["input_file"]) as csvfile:
    	data = csv.reader(csvfile, delimiter = ',')
    	for row in data:
            x.append(row[0])
            y.append(row[1])
    
    x=np.array(x[1:],dtype=np.float32)   
    y=np.array(y[1:],dtype=np.float32)   
    
    #Creating a matrix of each type to further compute
    matrix_y = y
    matrix_x = []

    for i in range(len(y)):
            matrix_x.append([x[i]*x[i],x[i],1])
    
    y = np.array(matrix_y)
    x = np.array(matrix_x)
    
    xcoords = []
    ycoords = matrix_y
    for xi in x:
    	# yi.append(xi[0]*B[0] + xi[1]*B[1] + xi[2]*B[2])
    	xcoords.append(xi[1])
    
    nPoints = len(y)
    pointSet = []
    max_inliers = 0
    
    # while max_inliers < 213:
    # This while statement is tailored to work best for this dataset
    
    for _ in range(N):
    
    	#Selecting three random points to generate the plot for the Parabola
    	pntsList = random.sample(range(len(matrix_y)), 3)
    	while Repeating_or_not(pntsList, pointSet):
    		pntsList = random.sample(range(len(matrix_y)), 3)
    	pointSet.append([pntsList])
    	mat_x = []
    	mat_y = []
    
    	for i in pntsList:
    		mat_y.append(matrix_y[i])
    		mat_x.append(matrix_x[i])
    
    
    	y = np.array(mat_y)
    	x = np.array(mat_x)
    
    	one = np.linalg.inv(x)
    	two = y
    
    	B = np.matmul(one, two)
    
    	curr_inliers = 0
    
    	for pointID in range(nPoints):
    
    		if check_whether_outlier_inlier(B[0], B[1], B[2], xcoords[pointID], ycoords[pointID], threshold):
    			curr_inliers += 1
    
    	if curr_inliers > max_inliers:
    		max_inliers = curr_inliers
    		best_model = [curr_inliers, B[0], B[1], B[2]]
    
    
    print(best_model)
    
    yi = []
    for xi in matrix_x:
    	yi.append(xi[0]*best_model[1] + xi[1]*best_model[2] + xi[2]*best_model[3])
    
    plt.plot(xcoords, matrix_y, 'bo')
    plt.plot(xcoords, yi, 'ro')
    plt.show()
    
    #print(pointSet)
