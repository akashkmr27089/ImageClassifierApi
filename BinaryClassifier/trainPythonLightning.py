import pytorch_lightning as pl
from Models import PretrainedNetwork
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from DatasetPytorchLightning import DatasetBinary

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class BinaryClassifierImage(pl.LightningModule):
    def __init__(self, model: nn.Module, lr:int):
        super(BinaryClassifierImage, self).__init__()
        self.model = model
        self.lr = lr
        
    def forward(self, x):
        y = self.model(x)
        return y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bs, _, _, _ = x.size()
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.reshape(len(y),1).float())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        bs, _, _, _ = x.size()
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.reshape(len(y),1).float())
        self.log('val_loss', loss, prog_bar=True)
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }
    
data = DatasetBinary()

ae_model = BinaryClassifierImage( PretrainedNetwork(4096, 1).model, 1e-4)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(gpus=1, max_epochs=25, amp_level='O2', precision=16, callbacks=[lr_monitor,EarlyStopping(monitor='val_loss')])

lr_finder = trainer.tuner.lr_find(ae_model, data)
lr_finder.results

fig = lr_finder.plot(suggest=True)
new_lr = lr_finder.suggestion()
ae_model.lr = new_lr

## Model Training 
trainer.fit(ae_model, data)