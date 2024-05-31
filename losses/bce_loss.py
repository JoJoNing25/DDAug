import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight).to(device)
        self.fn = torch.nn.BCEWithLogitsLoss(weight=weight)
    
    def forward(self, x, y):
        return self.fn(x, y)