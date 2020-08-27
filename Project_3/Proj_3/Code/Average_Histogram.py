#########################################################################
#Calculates Average Histogram of one buoy at a time
#This will give us the intuition of all the channels that will be used to 
#detect the buoys for 1D training.
#Also it gives us the peaks of pixel intesities.
#########################################################################

import glob
import cv2 
import numpy as np
from matplotlib import pyplot as plt

#set the path to the dataset of different coloured buoy each time
path = glob.glob(r"C:\Users\sukoo\673\Project3\Training_Data\Yellow_Training\*.jpg")

#initialise the list of final set of histogram
final_histr = []

#defining colours
color = ('b','g','r')
#iterating over each image from the path
for img in path: 
    #reading the image and pre-processing (gaussian blur)
    n = cv2.imread(img)
    n = cv2.GaussianBlur(n,(5,5),0)
    #initialise another empty list to store 3 lists of histogram values for blue, green, red respectively.
    histr_all=[]
    
    #'i' takes values of enumerate(colour)= (o,1,2)
    for i,col in enumerate(color):
        histr = cv2.calcHist([n],[i],None,[256],[0,256]) 
        #appends 3 vectors of 256 elements each to histr_all
        histr_all.append(histr)
    #converted the list(histr_all) to numpy array and flatten it so that it has 256 x 3 = 768 elements
    histr_all = np.array(histr_all, dtype = "uint8")
    histr_all=histr_all.flatten()
    #keep on adding all the rgb vector of 768 elements of every image in one 'coloured' buoy
    final_histr.append(histr_all)
    
final_histr = np.array(final_histr)
#final_histr is converted to np.array to apply mean function "COLUMN WISE" that's why 'axis =0' 
# 1 | 2 | 3 | 4 -i
# 5 | 6 | 7 | 8 -ii
#----------------------
# 3 | 4 | 5 | 6 - avg(i,ii)
mean_vec= np.mean(final_histr, axis =0)
#split the channels' mean values
mean_b = mean_vec[0:256]
mean_g = mean_vec[256:512]
mean_r = mean_vec[512:]
plt.plot(mean_b,color = 'b')
#plt.show()
plt.plot(mean_g,color = 'g')
#plt.show()
plt.plot(mean_r,color = 'r')
plt.show()
