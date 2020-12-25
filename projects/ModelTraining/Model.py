import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
# from Dataset import TestData

class Model:

    def __init__(self):
        # Using Pretrained AlexNet
        self.model = models.alexnet(pretrained = True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to('cuda')
        self.n_inputs = 4096
        self.n_classes = 100

    def summary(self):
        print(" Summary of the AlexNet Model :")
        print(summary(self.model, (3,224,224)))

    def ProcessingModel(self):
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Sequential(
                      nn.Linear(self.n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, self.n_classes),                   
                      nn.LogSoftmax(dim=1))
    
        #Verify if the model is Adjusted 
        try:
            predict = model(TestData.to(device))
            if len(predict[0]) != self.n_classes:
                raise Exception
        except:
            print("Exception : Number of Ouptut Mismatch")
        finally:
            pass
