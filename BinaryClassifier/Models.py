## Loading all the data 

import torch
from torch.autograd import Variable
from torch.nn import Linear, Sigmoid, ReLU, LogSoftmax, AdaptiveAvgPool2d, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import EnvConfig
import numpy as np
from torchvision import models
from torchsummary import summary

class Network(Module):
    
    def __init__(self, inputSize, outputSize):
        super(Network, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        
        self.avgpool = AdaptiveAvgPool2d(output_size=(6,6))
        
        self.classifier = Sequential(
            Linear(inputSize, 4096),
            ReLU(),
            Linear(4096, 256), 
            ReLU(), 
            Dropout(0.4),
            Linear(256, outputSize),                   
            Sigmoid()
            )

    def forward(self, data):
        x = self.features(data)
        x =  self.avgpool(x)
        x = x.reshape(EnvConfig.batch_size, -1)
        return self.classifier(x)
    
    
class PretrainedNetwork:
    
    def __init__(self, inputSize, outputSize, FreezeNetwork = False):
        self.model = models.alexnet(pretrained = True, progress = True)
        
        # if(FreezeNetwork):
        #     for param in self.model.parameters():
        #             param.requires_grad = False
        
        
        # self.model.classifier[6] =  Sequential(
        #     Linear(inputSize, 4096),
        #     Sigmoid(),
        #     Linear(4096, 256), 
        #     Sigmoid(), 
        #     Dropout(0.4),
        #     Linear(256, outputSize),                   
        #     Sigmoid()
        #     )
        self.model.classifier[-1] = Sequential(
            Linear(inputSize, 1),
            Sigmoid()
            )
        
        self.model.to(EnvConfig.device)
        
class PretrainedNetworkVGG:
    
    def __init__(self, inputSize, outputSize, FreezeNetwork = False):
        self.model = self.vgg16 = models.vgg16(pretrained=True)
    
        
        self.model.classifier[-1] = Sequential(
            Linear(inputSize, 1),
            Sigmoid()
            )
        
        self.model.to(EnvConfig.device)
        
        
    
    
    
# doc = Network(294912, 2)
# inputSize = np.prod([x for x in list(doc.features(torch.rand(EnvConfig.batch_size, 3, EnvConfig.ImageDim, EnvConfig.ImageDim)).shape)])