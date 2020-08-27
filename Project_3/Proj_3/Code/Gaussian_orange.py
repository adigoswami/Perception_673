import glob
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math

# function that gives gaussian probability 
def prob_gaussian(mean, var, x):
    denom = (2*math.pi*var**2)**(1/2)
    power = -0.5*((x - mean)/var)**2
    numerator = math.e**power
    
    return (numerator/denom)

# function to find new mean in each iteration
def new_mean(prob):
    SOP = 0
    SUM_Of_Prob = sum(prob)
    for i in range(len(prob)):
        n = prob[i]*add[i]
        SOP += n   
    return (SOP/SUM_Of_Prob)

# function to find new variance (standard deviation )
def new_variance(prob,mean):
    SOP = 0  #(sum of (product of( probability and (( data - mean )**2 ) ) ))
    SUM_of_Prob = sum(prob)
    for i in range(len(prob)):
        n = prob[i]*((add[i]-mean)**2)
        SOP += n
    return math.sqrt(SOP/SUM_of_Prob)

def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
    s1 = 1/(np.sqrt(2*np.pi*variance))
    s2 = np.exp(-(np.square(data - mean)/(2*variance)))
    return s1 * s2

# initialised three mean and variance(std. deviation) values for three gaussians  
mu1, var1 = 50, 10 
mu2, var2 = 100, 10
mu3, var3 = 150, 10 

# Set the path to the folder for the training dataset (Orange dataset for training)
path = glob.glob(r"C:\Users\sukoo\673\Project3\Training_Data\Orange_Training\*.jpg")
c =0 #iteration counter initialised
while(c!=50): #iterations to be run 50 times
    #initialised bayesian probability lists for gaussian 1, 2 and 3. In the end gives the final bayesian prob. for all images
    bayes1 = []
    bayes2 = []
    bayes3 = []
    # initialised add list. In the end gives the flat array list of all images (length of add == length of any baysian list)
    add = []
    for img in path: 
        #reading the image and pre-processing
        img = cv2.imread(img)
        #taking red channel out for orange colored buoy
        r_img = img[:,:,2].astype(np.float32)
        flat_array = list(r_img.flatten()) #flatten and convert into list
        add = add+flat_array #keep adding to add[] list
        for i in flat_array: #over every pixel value in flat_array
            prob_1 = prob_gaussian(mu1, var1, i) #find probability 1 for gaussian 1 using function : prob_gaussian(mean, variance, value in flat_array)
            prob_2 = prob_gaussian(mu2, var2, i) #find probability 1 for gaussian 2 using function : prob_gaussian(mean, variance, value in flat_array)
            prob_3 = prob_gaussian(mu3, var3, i) #find probability 1 for gaussian 3 using function : prob_gaussian(mean, variance, value in flat_array)
            
            #finding bayesian probability for gaussian 1
            eq1 = (prob_1 *(1/3))/(prob_1*(1/3)+prob_2*(1/3)+prob_3*(1/3))
            eq2 = (prob_2 *(1/3))/(prob_1*(1/3)+prob_2*(1/3)+prob_3*(1/3))
            eq3 = (prob_3 *(1/3))/(prob_1*(1/3)+prob_2*(1/3)+prob_3*(1/3))
            bayes1.append(eq1)
            bayes2.append(eq2)
            bayes3.append(eq3)

    #finding new mean and variance for gaussian 1
    mu1 = new_mean(bayes1)
    var1 = new_variance(bayes1,mu1)
    #finding new mean and variance for gaussian 2
    mu2 = new_mean(bayes2)
    var2 = new_variance(bayes2,mu2)
    #finding new mean and variance for gaussian 3
    mu3 = new_mean(bayes3)
    var3 = new_variance(bayes3,mu3)
    print(mu1,mu2,mu3)
    print(var1,var2, var3)
    print(c)
    c += 1 #incrementing counter c for iteration

x1 = np.random.normal(mu1, np.sqrt(var1), 100)
x2 = np.random.normal(mu2, np.sqrt(var2), 100)
x3 = np.random.normal(mu3, np.sqrt(var3), 100)
X = np.array(list(x1) + list(x2) + list(x3))
np.random.shuffle(X)
print("Dataset shape:", X.shape)
bins = np.linspace(np.min(X),np.max(X),100)

#Plotting the Gaussian for orange bouy
plt.figure(figsize=(10,7))
plt.xlabel("$x$")
plt.ylabel("pdf")
plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

plt.plot(bins, pdf(bins, mu1, var1), color='red', label="True pdf")
plt.plot(bins, pdf(bins, mu2, var2), color='red')
plt.plot(bins, pdf(bins, mu3, var3), color='red')

plt.legend()
plt.plot()
plt.savefig("orange.png")
plt.show()