from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class Dataset:

    def __init__(self):

        self.traindir = "D:/Research/CarControl/CarApi/projects/ModelTraining/dataset/Output/train"
        self.validdir = "D:/Research/CarControl/CarApi/projects/ModelTraining/dataset/Output/val"
        self.batch_size = 32

        # Image transformations
        self.image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            # Validation does not use augmentation
            'valid':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data = {
            'train':
            datasets.ImageFolder(root=self.traindir, transform=self.image_transforms['train']),
            'valid':
            datasets.ImageFolder(root=self.validdir, transform=self.image_transforms['valid']),
        }

        # Dataloader iterators, make sure to shuffle
        dataloaders = {
            'train': DataLoader(data['train'], batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(data['valid'], batch_size=self.batch_size, shuffle=True)
        }

        self.TrainIter = dataloaders['train']
        self.validIter = dataloaders['val']

    def TestData(self):
        for x in self.TrainIter:
            break
        return x[0][0].resize(1,3,224,224)
        
        


# Image transformations
# image_transforms = {
#     # Train uses data augmentation
#     'train':
#     transforms.Compose([
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.ColorJitter(),
#         transforms.RandomHorizontalFlip(),
#         transforms.CenterCrop(size=224),  # Image net standards
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])  # Imagenet standards
#     ]),
#     # Validation does not use augmentation
#     'valid':
#     transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# traindir = "C:/Users/AAKASH_Saggezza/Desktop/Testing OD/dataset/train"
# validdir = "C:/Users/AAKASH_Saggezza/Desktop/Testing OD/dataset/test"
# batch_size = 32


# # Datasets from folders
# data = {
#     'train':
#     datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
#     'valid':
#     datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
# }

# # Dataloader iterators, make sure to shuffle
# dataloaders = {
#     'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
#     'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
# }