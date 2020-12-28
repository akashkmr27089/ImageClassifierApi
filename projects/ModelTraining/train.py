from Model import Model

# #Usnig Pretrained Alexnet 
# model = models.alexnet(pretrained = True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to('cuda')

# # Summerise the Model
# # summary(model, (3, 224, 224))

# # Freeze model weights
# for param in model.parameters():
#     param.requires_grad = False

# n_inputs = 4096
# n_classes = 100

# # Add on classifier  #Classifier Model with the custom made classifier
# model.classifier[6] = nn.Sequential(
#                       nn.Linear(n_inputs, 256), 
#                       nn.ReLU(), 
#                       nn.Dropout(0.4),
#                       nn.Linear(256, n_classes),                   
#                       nn.LogSoftmax(dim=1))

# predict = model(TestData.to(device))

ModelClass = Model()
model = ModelClass.model
ModelClass.summary()
if(ModelClass.TestingNetwork() == True):
    print("Model Working Fine")
else:
    print("Some Problem in the Model")