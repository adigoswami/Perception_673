1. Extract the project folder.

2. Open Command prompt and navigate to the folder location

3. Run the program - Frame_extract.py by running the command "python Frame_extract.py"           
   (This extracts frames from the video at the path of your interpreter (or where the python file you executed is saved).
    Please change the path to your video file location in the python file "Frame_extract.py")

4. To find the average histogram of the dataset: 
   Run the file - Average_histogram.py by executing the command "python Average_histogram.py"
   (Please change the path of input data to the dataset folder of each buoy- green, orange, yellow.
   This file should be run 3 times for 3 types of buoys which gives 3 histograms.

   Now this gives intuition of channels to be used to detect buoys.

5. Next we go to train 1D Gaussian for each buoy in which expectation maximization gives us the converged value of means      and standard deviations for 3 clusters(gaussians).
   To achieve this we run the files : Gaussian_green.py
				      Gaussian_yellow.py
                                      Gaussian_orange.py
   by simply executing command "python Filename.py" , Filename can be  the 3 files above.
   (Please change the path to the respective dataset folder) 

6. Values obtained of Mean and Standard Deviation (of each channel ) from step five are then used in the final code file -    Buoy_detection.py
7. Run "python Buoy_detection.py" to run the video file for color segmentation and bouy detection for all three buoys.
   The Output will first be 3 different  GMM plots for the 3 buoys. Closing which will start the output video with the    detection.
   (Please change the path to the input video file). 

Please Note : Please Check the import libraries used in the python files are installed in your system.