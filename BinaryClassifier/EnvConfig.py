import os 
import torch


RootDir = "D:/Research/ML_Challanges/kaggle/ImageClassification/BineryClassification/DogsVsCats/"
trainDir = os.path.join(RootDir, 'dataset/train')
testDir = os.path.join(RootDir, 'dataset/test')
valDir = os.path.join(RootDir, 'dataset/val')

batch_size = 16
ImageDim = 224  #width and height of image 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'