import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
from Dataset import Dataset

class Model:

    def __init__(self):
        # Using Pretrained AlexNet
        self.model = models.alexnet(pretrained = True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = self.model.to(self.device)
        self.n_inputs = 4096
        self.n_classes = 102
        self.dataset = Dataset()
        
        # Freezing all the weights and then transfering to GPU
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Sequential(
                      nn.Linear(self.n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, self.n_classes),                   
                      nn.LogSoftmax(dim=1))
        
        # Transfering model to gpu
        self.model = self.model.to(self.device)

    def summary(self):
        print(" Summary of the AlexNet Model :")
        print(summary(self.model, (3,224,224)))

    def TestingNetwork(self):
        #Verify if the model is Adjusted 
        try:
            TempData = self.dataset.TestData().to(self.device)
            predict = self.model(TempData)
            if int(predict.size()[1]) != self.n_classes:
                raise Exception
        except:
            print("Exception : Number of Ouptut Mismatch")
            return False
        finally:
            return True
