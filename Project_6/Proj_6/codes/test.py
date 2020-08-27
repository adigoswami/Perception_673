# import all dependencies
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from torch.autograd import Variable
import os
from torchvision import datasets, models, transforms
from time import time
import copy
import math
import re
import shutil
import csv
from PIL import Image
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SUBMISSION_FILE = 'submission.csv'

class VGG16(nn.Module):

    def __init__(self, num_classes):
        """
        Convulational Neural Network that replicates VGG16 architecture
        Out_features =  num_classes (2)

        """
        super(VGG16, self).__init__()
        self.CNN = nn.Sequential(
            #First CNN block
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            #Second CNN Block
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            #Third Block
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            #Fourth Block
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            #Fifth Block
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.linear = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
            # nn.LogSoftmax(dim=1) #To limit the output from 0 to 1
        )

    def forward(self, x):
        x = self.CNN(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224,224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out.reshape((1,3,224,224))

def test_data_from_fname(fname):
    im = Image.open(f'{test_data_dir}/{fname}')
    return apply_test_transforms(im)

def extract_file_id(fname):
    # print("Extracting id from " + fname)
    return int(re.search('\d+', fname).group())




test_data_dir = r"C:/Users/sukoo/673/Project6/dogs-vs-cats/data/test"
test_data_files = os.listdir(test_data_dir)

def main():
    # model = VGG16(num_classes = 2).to(device)
    model = VGG16(num_classes = 2)
    model.load_state_dict(torch.load('vgg16.pt'))
    print("Testing...")
    test(model)

def test(model):
    start = time()
    flag = False
    model.eval()
    for fname in test_data_files:

        image = test_data_from_fname(fname)
        tag = extract_file_id(fname)
        # print("Image ID", tag)
        with torch.no_grad():

            image.to(device)
            # print(image.shape)
            output = model(image).squeeze()

            if output >= 0.5: 
                label = 1
            else:
                label = 0

            with open(SUBMISSION_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                if flag is not True:
                    writer.writerow(["SID", "Label (1: Dog, 0: Cat)"]) # writeline(frame id/name, label)
                    writer.writerow([tag, label])
                    flag = True
                else:
                    writer.writerow([tag, label])
    timeElapsed = time() - start
    print(f'Training complete in {timeElapsed//60:.0f}m {timeElapsed%60:.0f}s')


if __name__ == '__main__':
    main()
