This Project is done on Pytorch Platform with Cudatoolkit 10.2
You can either use google colab or a local machine with or without a GPU to run the Project and Train/Test the model.

0. Download the submission folder and place the dataset in this directory (in the format as explained in step 3) 
1. Install Pytorch in a local environemt (preferably on Anaconda) by running this command( for windows):
   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

2. Download the necessary libaraies and dependancies to get the project up and running	 
   torchvision, csv, PIL
3. Before Training and testing the model, please setup the dataset folder in the following structure:
   data -
	train - 
		cats (Place all the 12500 cats files from the original train folder here) 
		dogs (Place all the 12500 dogs files from the original train folder here)
	val - (create this empty folder)
		cats (create this empty folder)
		dogs (create this empty folder)
	test - (contains the 12500 test images)

**Note : to complete the dataset formation please move 3200 images from 
4. if running on your local machine, Navigate to this folder in your command prompt
5. Change the paths in the python files according to the path of the downloaded submission foleder on ypu machine.
   (replace the data_dir,train_dir, val_dir, train_dogs_dir, train_cats_dir, val_dogs_dir, val_cats_dir) in the train.py
   (replace the test_data_dir) in the test.py
6. First to setup the dataloaders and  train and save the model, please run the commad 
   python train.py

Note : Crosscheck that batch size = 10, epoch = 16

7. we have our model saved by the name 'vgg16.pt'
8. Now to test out model with the test dataset, run:
   python test.py
9. A submission.csv file will be saved as our final output.

10. If running on google colab create a notebook and copy paste the content of both the python files in the submission folder in that notebook.

Note : Please be sure that the required dependancies are completely installed and setup vefore running this project.

