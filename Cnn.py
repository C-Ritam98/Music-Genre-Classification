# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 17:45:10 2021

@author: RITAM
"""

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import librosa

device = torch.device('cuda')
import librosa.display

#%%
# identify the class of the instance


# function to evaluate the model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)



# directory that holds the dataset
directory = "./genres/"
f = open("my_cnn_mel.dat", 'wb')

i = 0


for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):
        y, sr = librosa.load(directory+folder+'/'+file, 22050)
        S = librosa.stft(y, n_fft=1024, hop_length=512, win_length=1024)
        mel_basis = librosa.filters.mel(22050, n_fft=1024, n_mels=128)
        mel_S = np.dot(mel_basis, np.abs(S))
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.T
        feature = (mel_S, i)
        pickle.dump(feature, f)

f.close()

#%%


dataset = []
def loadDataset(filename, split,trSet, teSet):
    with open(filename,'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
        
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])
  

trainingSet = []
testSet = []

loadDataset("my_cnn_mel.dat", 0.75, trainingSet, testSet)

X_train = []
X_test = []
y_train = []
y_test = []

'''for i in trainingSet:
    temp = []
    for j in i[0]:
        for k in j:
            temp.append(list(j))
    #temp = np.array(temp)
    X_train.append(temp)
    y_train.append(i[1])
for i in testSet:
    temp = []
    for j in i[0]:
        temp.append(list(j))
    temp = np.array(temp)
    X_test.append(temp)
    y_test.append(i[1])


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)   
y_test = np.array(y_test)'''


#%%
class Cnn(nn.Module):
    
    def __init__(self):
        super(Cnn, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        '''self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))'''
        
        self.fc1 = nn.Linear(in_features=323*32*31, out_features=1000)
        self.drop = nn.Dropout2d(0.25)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=10)
        
    def forward(self, x):
        # x : [1,2997,13]
        out1 = self.layer1(x)
        # out1 : [1,1500,8]*8
        out2= self.layer2(out1)
        # out2 : [1,749,3]*16
        out3 = out2.view(out2.size(0), -1)
        #out4 = [1,749*16*3]
        lin1 = self.fc1(out3)
        lin1 = self.relu(lin1)
        lin1 = self.drop(lin1)
        lin2 = self.fc2(lin1)
        lin2 = self.relu(lin2)
        lin2 = self.drop(lin2)
        lin3 = self.fc3(lin2)
        lin3 = self.relu(lin3)
        lin3 = self.drop(lin3)
        lin4 = self.fc4(lin3)
        lin4 = self.soft(lin4)
        
        return lin4
    
    
if __name__ == '__main__':
    
    model = Cnn()
    model.to(device)
    
    error = nn.CrossEntropyLoss()
    
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_epochs =  1
    train_loss = []
    
    
    for epoch in range(num_epochs):
        batch_loss= 0 
        for images, labels in trainingSet[:1]:
            
            images, labels = torch.as_tensor(images,dtype=torch.float), torch.as_tensor(labels).unsqueeze(0)
            images = torch.cat((images/100,torch.zeros((3,128))),0).unsqueeze(0).unsqueeze(0)
            model.train()
            
            images, labels = images.to(device), labels.to(device)
            # images : [100,1,28,28]
            # labels : [100]
            
            # Forward pass 
            outputs = model(images)
            loss = error(outputs, labels)
            batch_loss+=loss
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            #Propagating the error backward
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()
        print("No of epoch: {}, Loss: {}".format(epoch, batch_loss.data))
        train_loss.append(batch_loss.item())
        
    torch.save(model.state_dict(),'cnn_for_music_classification.pt')

    # ploting training loss per epochs
    
    plt.title('training loss per Epoch')
    plt.plot(train_loss,color='black')
    plt.xlabel('Epochs')
    plt.ylabel('trainning loss')
    plt.show()
    
    
    
#%%

