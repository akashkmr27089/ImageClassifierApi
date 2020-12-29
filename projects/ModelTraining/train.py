from Model import Model
from Dataset import Dataset
import torch
from torch import optim
import torch.nn as nn

#Variables 
epochs = 3

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
    for i in range(epochs):
        avgLossPerEpochs = 0.0
        iterValuePerEpochs = 0
        TrainIterDataset = iter(dataset.dataloaders['train'])
        for features, labels in TrainIterDataset:
            #One Hot Labels 
            labels_onehot = torch.zeros(len(labels), 102).scatter_(1, labels.unsqueeze(1), 1)
            iterValuePerEpochs += 1
            optimizer.zero_grad()
            output = model(features.to(device))
            loss = criterion(output, labels.to(device))
            avgLossPerEpochs += float(loss)
            print(loss,iterValuePerEpochs)
            loss.backward()
            optimizer.step()
            avgLossPerEpochs = avgLossPerEpochs/iterValuePerEpochs
        print(" For epoch {} average loss is {}".format(i, avgLossPerEpochs))
        
except RuntimeError:
    print("Runtime Error Occured ")
    print(labels_onehot, labels_onehot.size())
    print(len(labels), labels.unsqueeze(1))

# feature, labels = dataset.NextTrain()
# predict = model(feature.to(device))
# print('Predicted ')
# print(predict)