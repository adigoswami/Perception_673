import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
import math
import copy
counter = 0

##############################################################################
################  gaussian: returns y-cordinates of gaussian  ################
################  Input: x(numpy array), mean(), var()        ################
##############################################################################
def gaussian(x, mean, var):
    return ((1/(var*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mean, 2.) / (2 * np.power(var, 2.))))

x=list(range(0, 256)) # x-coordinate of all the Gaussian 

# Trained parameters for three Gaussian for the orange buoy
red_parameters = [np.array([169.1073862]), np.array([37.1076208]), np.array([240.5153214]), np.array([7.4222478]), np.array([252.8131563]), np.array([2.0198301])]
# Trained parameters for three Gaussian for the green buoy
green_parameters = [np.array([170.01956154923656]), np.array([35.06322917552536]), np.array([237.13832679215125]), np.array([8.583996207342343]), np.array([246.8302005770773]), np.array([4.8911086062220095])]
# Trained parameters for three Gaussian for the yellow buoy
yellow_parameters = [np.array([182.11780533350807]), np.array([29.660953276744305]), np.array([233.56093890526682]), np.array([5.890672397900842]), np.array([234.86433392497744]), np.array([1.9116253637546492])]

redGaussian1 = gaussian(x, red_parameters[0], red_parameters[1]) # 1st 1-D Gaussian in the red channel for the orange buoy
redGaussian2 = gaussian(x, red_parameters[2], red_parameters[3]) # 2nd 1-D Gaussian in the red channel for the orange buoy
redGaussian3 = gaussian(x, red_parameters[4], red_parameters[5]) # 3rd 1-D Gaussian in the red channel for the orange buoy

greenGaussian1 = gaussian(x, green_parameters[0], green_parameters[1]) # 1st 1-D Gaussian in the green channel for the green buoy
greenGaussian2 = gaussian(x, green_parameters[2], green_parameters[3]) # 2nd 1-D Gaussian in the green channel for the green buoy
greenGaussian3 = gaussian(x, green_parameters[4], green_parameters[5]) # 3rd 1-D Gaussian in the green channel for the green buoy

yellowGaussian1 = gaussian(x, yellow_parameters[0], yellow_parameters[1]) # 1st 1-D Gaussian in the (green+red)/2 channel for the yellow buoy
yellowGaussian2 = gaussian(x, yellow_parameters[2], yellow_parameters[3]) # 2nd 1-D Gaussian in the (green+red)/2 channel for the yellow buoy
yellowGaussian3 = gaussian(x, yellow_parameters[4], yellow_parameters[5]) # 3rd 1-D Gaussian in the (green+red)/2 channel for the yellow buoy

print('1-D Gaussian for the red buoy')
plt.plot(redGaussian1, 'b') # Plotting 1st 1-D Gaussian in the red channel for the orange buoy with blue color
plt.plot(redGaussian2, 'g') # Plotting 2nd 1-D Gaussian in the red channel for the orange buoy with green color
plt.plot(redGaussian3, 'r') # Plotting 3rd 1-D Gaussian in the red channel for the orange buoy with red color
plt.show()

print('1-D Gaussian for the green buoy')
plt.plot(greenGaussian1, 'b') # Plotting 1st 1-D Gaussian in the green channel for the green buoy with blue color
plt.plot(greenGaussian2, 'g') # Plotting 2nd 1-D Gaussian in the green channel for the green buoy with green color
plt.plot(greenGaussian3, 'r') # Plotting 3rd 1-D Gaussian in the green channel for the green buoy with red color
plt.show()

print('1-D Gaussian for the yellow buoy')
plt.plot(yellowGaussian1, 'b') # Plotting 1st 1-D Gaussian in the (green+red)/2 channel for the yellow buoy with blue color
plt.plot(yellowGaussian2, 'g') # Plotting 2nd 1-D Gaussian in the (green+red)/2 channel for the yellow buoy with green color
plt.plot(yellowGaussian3, 'r') # Plotting 3rd 1-D Gaussian in the (green+red)/2 channel for the yellow buoy with red color
plt.show()

#CHANGE THE PATH ACCORDING TO YOUR VIDEO FILE LOCATION
cap = cv2.VideoCapture(r"C:\Users\sukoo\673\Project3\detectbuoy.avi") # Input Video
capout = cv2.VideoWriter('./GMMfitting.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (640,480)) # Output video

while (True): # Looping over the frames of the input video
    
    ret, image = cap.read() 
        
    if not ret: # If no frame is generated or the video has ended
        cv2.destroyAllWindows() # Destroy all Windows
        cap.release() # Releases software/hardware resource
        capout.release() # Releases video writing resource
        break
   
    image_r = image[:,:,2] # Red channel of the frame
    image_g = image[:,:,1] # Green channel of the frame
    image_b = image[:,:,0] # Blue channel of the frame
    
    img_out_r = np.zeros(image_r.shape, dtype = np.uint8) # Temporary frame for orange buoy
    img_out_g = np.zeros(image_r.shape, dtype = np.uint8) # Temporary frame for green buoy 
    img_out_y = np.zeros(image_r.shape, dtype = np.uint8) # Temporary frame for yellow buoy
        
    for i in range(0,image_r.shape[0]): # Looping over the length of the frame
        for j in range(0,image_r.shape[1]): # Looping over the width of the frame
            
            # Probabilities for orange buoy
            if redGaussian3[image_r[i][j]] > 0.15 and image_b[i][j] < 160: 
                img_out_r[i][j] = 255
                    
            if redGaussian2[image_r[i][j]] > 0.02 and image_b[i][j] < 160:
                img_out_r[i][j] = 0
                
            if redGaussian1[image_r[i][j]] > 0.001 and image_b[i][j] < 160:
                img_out_r[i][j] = 0
            
            # Probabilities for green buoy
            if image_r[i][j] < 200:
                if greenGaussian3[image_g[i][j]] > 0.0357 and greenGaussian2[image_g[i][j]] < 0.0299 and greenGaussian1[image_g[i][j]] < 0.0299:     
                    img_out_g[i][j]=255
                else:
                    img_out_g[i][j]=0
                    
            # Probabilities for yellow buoy
            if ((yellowGaussian3[image_r[i][j]] + yellowGaussian3[image_g[i][j]])/2) > 0.03  and ((yellowGaussian1[image_r[i][j]] + yellowGaussian1[image_g[i][j]])/2) < 0.015:
                img_out_y[i][j]=255
            else:
                img_out_y[i][j]=0
            
    # Morphological and Contouring Operations on the orange buoy
    ret, threshold_o = cv2.threshold(img_out_r, 240, 255, cv2.THRESH_BINARY) 
    kernel_o = np.ones((2,2),np.uint8) # Kernel for dilation
    dilation_o = cv2.dilate(threshold_o,kernel_o,iterations = 6)
    contours_o, _= cv2.findContours(dilation_o, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours_o:
        if cv2.contourArea(contour) <  3000 and cv2.contourArea(contour) >  300:
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x)-3,int(y)-4)
            radius = int(radius)
            if radius > 13:
                cv2.circle(image,center,radius,(0,0,255),2)
    

    
    # Morphological and Contouring Operations on the green buoy
    ret, threshold_g = cv2.threshold(img_out_g, 240, 255, cv2.THRESH_BINARY)
    kernel_g = np.ones((2,2),np.uint8) # Kernel for dilation
    dilation_g = cv2.dilate(threshold_g,kernel_g,iterations =9)
    contours_g, _= cv2.findContours(dilation_g, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours_g:
        if (cv2.contourArea(contour) <  3000 and cv2.contourArea(contour) >  300 and cv2.contourArea(contour) != 315.5 and cv2.contourArea(contour) != 309.5 and cv2.contourArea(contour) != 2868.5) :

            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x)-5,int(y))
            radius = int(radius)
            if radius > 13 and radius < 15.5:
                cv2.circle(image,center,radius,(0,255,0),2)
                break
    

          
    # Morphological and Contouring Operations on the yellow buoy
    ret, threshold = cv2.threshold(img_out_y, 240, 255, cv2.THRESH_BINARY)
    kernel1 = np.ones((2,2),np.uint8) # Kernel for Erosion
    erosion = cv2.erode(threshold,kernel1,iterations = 1)
    kernel2 = np.ones((50,50),np.uint8) # Kernel for closing
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2)
    erosion = cv2.erode(closing,kernel1,iterations = 5)
    contours, _= cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    


    for contour in contours:
        
        if (cv2.contourArea(contour) <  3000 and cv2.contourArea(contour) >  2000) or (cv2.contourArea(contour) <  3000 and cv2.contourArea(contour) >  30):

            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            if center[1] < 400 and center[1] > 100 and radius > 8:
                cv2.circle(image,center,radius,(0,255,255),2)
                break

#Uncomment for testing 
    # cv2.imshow('GMM_Fitting', image)
    # cv2.imshow('binary', closing)
    # #capout.write(image)
    # if cv2.waitKey(1) == 27: # Press 'ESC' to stop the processing and break out of the loop 
    #     cv2.destroyAllWindows() # Destroys all window after pressing 'ESC'
    #     #capout.release()
    #     cap.release() # Releases software/hardware resource 

#Comment for testing
    cv2.imshow('GMM_Fitting', image)
    #cv2.imwrite("%d.jpg" % counter, image)
    counter = counter + 1
    capout.write(image)
    if cv2.waitKey(1) == 27: # Press 'ESC' to stop the processing and break out of the loop 
        cv2.destroyAllWindows() # Destroys all window after pressing 'ESC'
        capout.release()
        cap.release() # Releases software/hardware resource