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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Compute Hardware: {device}')

train_dir = r'C:\Users\sukoo\673\Project6\dogs-vs-cats\data\train'
val_dir = r"C:\Users\sukoo\673\Project6\dogs-vs-cats\data\val"


train_dogs_dir = r'C:\Users\sukoo\673\Project6\dogs-vs-cats\data\train\dogs'
train_cats_dir = r'C:\Users\sukoo\673\Project6\dogs-vs-cats\data\train\cats'
val_dogs_dir = r'C:\Users\sukoo\673\Project6\dogs-vs-cats\data\val\dogs'
val_cats_dir = r'C:\Users\sukoo\673\Project6\dogs-vs-cats\data\val\cats'


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
batchSize = 10 #20
data_dir = r"C:\Users\sukoo\673\Project6\dogs-vs-cats\data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=8) for x in ['train', 'val']} #num_workers = 16
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(class_names) # => ['cats', 'dogs']
# print(f'Train image size: {dataset_sizes["train"]}')
# print(f'Validation image size: {dataset_sizes["val"]}')

# --------------------------------------------------------------
def main():
    model = ConvNet(num_classes = 2).to(device)

    epochs = 16 #25
    learningRate = 0.001
    momentum = 0.9
    weightDecay = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # weight_decay=weightDecay
    learningRateScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    error = torch.nn.BCELoss() # torch.nn.CrossEntropyLoss()
    model = train(model, error, optimizer, learningRateScheduler, epochs)
    torch.save(model.state_dict(), 'vgg16.pt')

def train(model, error, optimizer, scheduler, no_epochs=50):
    # losses = {'train': [], 'val': []}
    # accuracies = {'train': [], 'val': []}
    bestModel = copy.deepcopy(model.state_dict())
    bestAccuracy = 0.0

    start = time()
    for epoch in range(1,no_epochs+1):
        print(f'Epoch {epoch}/{no_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            elif phase == 'val':
                model.eval()   

            datasetLoss = 0.0
            running_corrects = 0
            # iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    # outputs = model(inputs).reshape((batchSize,))
                    outputs = model(inputs).squeeze()
                    loss = error(outputs, labels)
                    ## binary classification predictions
                    predictions = outputs.clone()
                    predictions[predictions >= 0.5] = 1
                    predictions[predictions < 0.5] = 0
                    # backward pass
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                datasetLoss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            # apply learning rate decay policy
            if phase == 'train':
                scheduler.step()

            # compute evaluation metrics
            datasetLoss /= dataset_sizes[phase]
            datasetAccuracy = running_corrects.double() / dataset_sizes[phase]
            ## log results
            # losses[phase].append(datasetLoss)
            # accuracies[phase].append(datasetAccuracy)
            print(f'{phase} Loss: {datasetLoss:.4f} | Acc: {datasetAccuracy:.4f}')

            # save model with best val accuracy
            if phase == 'val' and datasetAccuracy > bestAccuracy:
                bestAccuracy = datasetAccuracy
                bestModel = copy.deepcopy(model.state_dict())

    timeElapsed = time() - start
    print(f'Training complete in {timeElapsed//60:.0f}m {timeElapsed%60:.0f}s')
    model.load_state_dict(bestModel)

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    # ax1.plot(losses['train'], color='blue', label='Train Loss')
    # ax1.plot(losses['val'], color='red', label='Val Loss')
    # ax1.legend()
    # ax1.set(ylabel = 'loss')
    # ax1.set(xlabel = 'epoch')
    # # ax1.set_title('loss vs epoch')
    # ax2.plot(accuracies['train'], color='blue', label='Train Acc')
    # ax2.plot(accuracies['val'], color='red', label='Val Acc')
    # ax2.legend()
    # ax2.set(ylabel ='accuracy')
    # ax2.set(xlabel = 'epoch')
    # # ax2.set_title('accuracy vs epoch')

    # plt.show()

    return model

class ConvNet(nn.Module):

    def __init__(self, num_classes):
        """
        Convulational Neural Network that replicates VGG16 architecture
        Out_features =  num_classes (2)

        """
        super(ConvNet, self).__init__()
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
            # nn.LogSoftmax(dim=1) #To limit the output from 0 to 1 #use nll loss
        )

    def forward(self, x):
        x = self.CNN(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    main()