# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:13:07 2021

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
from sklearn.metrics import confusion_matrix,classification_report
import math
import matplotlib.pyplot as plt
import seaborn as sns

# function to get the distance between feature vecotrs and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k) 
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors

#%%
# identify the class of the instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

    return sorter[0][0]


# function to evaluate the model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)


#%%
# directory that holds the dataset
directory = "./genres/"
f = open("my.dat", 'wb')

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):
        (rate, sig) = wav.read(directory+folder+"/"+file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

#%%

dataset = []
def loadDataset(filename, split,trSet, teSet):
    with open("my.dat", 'rb') as f:
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

loadDataset("my.dat", 0.75, trainingSet, testSet)

#%%

def distance(instance1 , instance2 , k ):
    distance = 0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance
#%%
# making predictions using KNN
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 3)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)

#%%
y_true = []
y_pred = np.array(predictions)
for i in testSet:
    y_true.append(i[-1])
y_true = np.array(y_true)

cm  = confusion_matrix(y_true,y_pred)
    
print(cm)

sns.heatmap(cm,annot=True)
plt.title('confution matrix plot using KNN')
print("\nconfution matrix plot")
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.legend()
plt.show()
plt.savefig('cnn_cm_ritam.png')

print(classification_report(y_true,y_pred))
