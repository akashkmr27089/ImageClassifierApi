from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import EnvConfig
import os

class Dataset:

    def __init__(self):

        ##Initialising the variables 
        self.traindir = EnvConfig.trainDir
        self.validdir = EnvConfig.valDir
        self.testdir = EnvConfig.testDir
        self.batch_size = EnvConfig.batch_size
        self.TestImageLabels = os.listdir(self.testdir)
        self.curr_test_iter = -1

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
            
        self.testFun = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Loading Dataset from the folder 
        data = {
            'train':
            datasets.ImageFolder(root=self.traindir, transform=self.image_transforms['train']),
            'valid':
            datasets.ImageFolder(root=self.validdir, transform=self.image_transforms['valid']),
        }

        # Dataloader iterators, make sure to shuffle
        self.dataloaders = {
            'train': DataLoader(data['train'], batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(data['valid'], batch_size=self.batch_size, shuffle=True),
            
        }


        # Making it iterator on the go
        self.TrainIter = iter(self.dataloaders['train'])
        self.ValidIter = iter(self.dataloaders['val'])


    # def TestDataLoadersIter(self, batch_size, initialize = False):
    #     if(batch_size > len(self.TestImageLabels)):
    #         return False
    #     if(initialize):
    #         self.curr_test_iter = -1
    #     self.curr_test_iter += 1
    #     if(self.curr_test_iter*batch_size < len(self.TestImageLabels)):
    #         if( (self.curr_test_iter + 1)*batch_size >= len(self.TestImageLabels)):
    #             return self.TestImageLabels[self.curr_test_iter*batch_size: ]
    #         else:
    #             return self.TestImageLabels[self.curr_test_iter*batch_size: (1 + self.curr_test_iter)*batch_size]
    #     else:
    #         return self.TestDataLoadersIter(batch_size, True)
        
    def TestDataLoadersIter(self, batch_size, initialize = False):
        if(batch_size > len(self.TestImageLabels)):
            return False
        if(initialize):
            self.curr_test_iter = 0
        if(self.curr_test_iter < len(self.TestImageLabels)):
            if(self.curr_test_iter + batch_size >= len(self.TestImageLabels)):
                temp = self.TestImageLabels[self.curr_test_iter : ]
                self.curr_test_iter = batch_size - (len(self.TestImageLabels) - self.curr_test_iter)
                temp2 = self.TestImageLabels[0: self.curr_test_iter]
                return temp + temp2
            else:
                temp = self.TestImageLabels[self.curr_test_iter : self.curr_test_iter + batch_size]
                self.curr_test_iter += batch_size
                return temp
        else:
            return self.TestDataLoadersIter(batch_size, True)

    def TestDataImageIter(self):
        '''
        Function Returns Images in the batch

        '''
        outLabels = self.TestDataLoadersIter(self.batch_size)
        for i in outLabels:
            imageLoc = os.path.join(EnvConfig.testDir, i)
            
        
        
dog = Dataset()       

    # def NextTest(self):
    #     # Returning Test Data
    #     ## Will be a list containing (Feature, Labels )
    #     return next(self.TestIter)

    # def NextTrain(self):
    #     # Returning Train Data 
    #     ## Will be a list containing (Feature, Labels )
    #     return next(self.TrainIter)

    # def NextValidation(self):
    #     # Returning Validation Data 
    #     ## Will be a list containing (Feature, Labels )
    #     return next(self.ValidIter)