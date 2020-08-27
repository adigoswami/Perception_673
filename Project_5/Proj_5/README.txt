Before you try to run the Project_5_Solution.py file, make sure following dependencies are there:

1.OpenCV and compatible SIFT packages are installed along side your python environment.

2. If not, run following commands to install them:
	
	2.1) pip install opencv-python==3.4.2.16
	
	2.2) pip install opencv-contrib-python==3.4.2.16
	
	2.3) pip install scipy
	
	2.4) pip install matplotlib

3. Now make sure ReadCameraModel.py file is present in the same directory as the Project_5_Solution.py code file is.

4. Similarly UndistortImage.py file should be in the same directory as the Project_5_Solution.py code file is.

5. Give the address of the dataset images i.e. "stereo/centre/" to the variable-> 'path'. (Give relative path if the 'stereo' is 
   in the same directory as is the code or give the full path).

6. Again give the path of the camera model files (in the folder-> 'model', already a relative path is given in the code use it 
   if the 'model' folder is in the same directory as is the code else provide full path).

7. Link to videos: https://drive.google.com/drive/folders/1i97sx_gKZooNdgkXyjW56FEUp7l227Pz?usp=sharing