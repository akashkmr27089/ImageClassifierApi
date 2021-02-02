from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import EnvConfig
import pytorch_lightning as pl
from typing import Dict, Callable, Optional, Any, Tuple
import multiprocessing
from PIL import Image
import os 
import torch

class TestDataLoader(Dataset):
    def __init__(self, path: str):
        
        self.img_paths = os.listdir(path)
        self.classes = [int(img_path.split(".")[0]) for img_path in self.img_paths]
        self.img_paths = [os.path.join(path, img_path) for img_path in self.img_paths]
            
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_paths[index])
        dataTransform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor_img = dataTransform(img)
        return tensor_img, torch.tensor(self.classes[index], dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.img_paths)

class DatasetBinary(pl.LightningDataModule):
    
    def __init__(self):
        super().__init__()
        self.test_dir = EnvConfig.testDir
        self.train_dir = EnvConfig.trainDir
        self.valid_dir = EnvConfig.valDir
         
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
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
            'val':
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
            ])
        }
            
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.image_transforms['train'])
            self.val_dataset = datasets.ImageFolder(root=self.valid_dir, transform=self.image_transforms['val'])
        if stage == 'test' or stage is None:
            self.test_dataset = TestDataLoader(self.test_dir)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=EnvConfig.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=EnvConfig.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=EnvConfig.batch_size, shuffle=False)



