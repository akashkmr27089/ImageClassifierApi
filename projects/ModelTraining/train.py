'''
The Following contains 
'''

from Model import Model
from Dataset import Dataset
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt

#Variables 
epochs = 5
PrintEpoch = 10
PrintEpochValidation = 20

# Loading Model 
ModelClass = Model()
model = ModelClass.model
ModelClass.summary()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Testing Model
if(ModelClass.TestingNetwork() == True):
    print("Model Working Fine")
else:
    print(" Some Problem in Model Pre Model Testing")
    raise Exception("PreTesting Error")

# Loading Database
dataset = Dataset()

#Defining Optimisers and BackPropogation Loss Function 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Loopoing through entire dataset 
try:
    LossValue = 0.0
    trainLoss = []
    validationLoss = []
    for i in range(epochs):

        avgLossPerEpochs = 0.0
        iterValuePerEpochs = 0
        avgValLossPerEpochs = 0.0
        iterValuePerEpochsValidataion = 0

        TrainIterDataset = iter(dataset.dataloaders['train'])
        VlidationIterDataset = iter(dataset.dataloaders['val'])

        ## Training for each epochs 
        for features, labels in TrainIterDataset:
            #One Hot Labels 
            # labels_onehot = torch.zeros(len(labels), 102).scatter_(1, labels.unsqueeze(1), 1)
            iterValuePerEpochs += 1
            optimizer.zero_grad()
            output = model(features.to(device))
            loss = criterion(output, labels.to(device))
            avgLossPerEpochs += float(loss)
            loss.backward()
            optimizer.step()
            if(iterValuePerEpochs % PrintEpoch == 0):
                print('Loss at Epoch {} Iteration {} is {}'.format(i + 1, iterValuePerEpochs, float(loss)))
        avgLossPerEpochsFinal = avgLossPerEpochs/iterValuePerEpochs
        trainLoss.append(avgLossPerEpochsFinal)
        print(avgLossPerEpochs, iterValuePerEpochs)

        
        ## Validation Model 
        print("Validation for epoch {} Started ".format(i+1))
        for features, labels in VlidationIterDataset:
            iterValuePerEpochsValidataion += 1
            output = model(features.to(device))
            loss = criterion(output, labels.to(device))
            avgValLossPerEpochs += float(loss)
            if(iterValuePerEpochsValidataion % PrintEpochValidation == 0):
                print(' Validation Loss at Epoch {} Iteration {} is {}'.format(i + 1, iterValuePerEpochsValidataion, float(loss)))
        avgValLossPerEpochsFinal = avgValLossPerEpochs/iterValuePerEpochsValidataion
        print(avgValLossPerEpochs, iterValuePerEpochsValidataion)
        print(" For epoch {} average loss is {} with validation Error {}".format(i + 1, avgLossPerEpochsFinal, avgValLossPerEpochsFinal))
        validationLoss.append(avgValLossPerEpochsFinal)
    
    #Plotting for training and Validation
    plt.plot(trainLoss)
    plt.plot(validationLoss)
    plt.show()

except RuntimeError:
    print("Runtime Error Occured ")
    print(labels_onehot, labels_onehot.size())
    print(len(labels), labels.unsqueeze(1))


