from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class Dataset:

    def __init__(self):

        self.traindir = "D:/Research/CarControl/CarApi/projects/ModelTraining/dataset/Output/train"
        self.validdir = "D:/Research/CarControl/CarApi/projects/ModelTraining/dataset/Output/val"
        self.testdir = "D:/Research/CarControl/CarApi/projects/ModelTraining/dataset/Output/test"
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
            'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Loading Dataset from the folder 
        data = {
            'train':
            datasets.ImageFolder(root=self.traindir, transform=self.image_transforms['train']),
            'valid':
            datasets.ImageFolder(root=self.validdir, transform=self.image_transforms['valid']),
            'test':
            datasets.ImageFolder(root=self.testdir, transform=self.image_transforms['test'])
        }

        # Dataloader iterators, make sure to shuffle
        dataloaders = {
            'train': DataLoader(data['train'], batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(data['valid'], batch_size=self.batch_size, shuffle=True),
            'test': DataLoader(data['test'], batch_size=self.batch_size, shuffle=True)
        }

        # Making it iterator on the go
        self.TrainIter = iter(dataloaders['train'])
        self.ValidIter = iter(dataloaders['val'])
        self.TestIter = iter(dataloaders['test'])


    def TestData(self):
        # for x in self.TrainIter:
        #     break
        # return x[0][0].resize(1,3,224,224)
        feature, _ = next(self.TrainIter)
        return feature[0].resize(1,3,224,224)

    def NextTest(self):
        # Returning Test Data
        ## Will be a list containing (Feature, Labels )
        return next(self.TestIter)

    def NextTrain(self):
        # Returning Train Data 
        ## Will be a list containing (Feature, Labels )
        return next(self.TrainIter)

    def NextValidation(self):
        # Returning Validation Data 
        ## Will be a list containing (Feature, Labels )
        return next(self.ValidIter)
        