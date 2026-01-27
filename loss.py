# %% 
import torch
from torch import nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mse = self.mse(predictions, targets)
        return  mse