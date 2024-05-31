import mlconfig
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from losses.ntxent import NTXentLoss
from losses.ntxent_kornia import NTXentLossKornia
from losses.ntxent_lidaug import NTXentLossLIDAug
from losses.bce_loss import BCELoss
mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(torch.nn.MSELoss)
mlconfig.register(NTXentLoss)
mlconfig.register(torch.nn.BCEWithLogitsLoss)
mlconfig.register(NTXentLossKornia)
mlconfig.register(NTXentLossLIDAug)
mlconfig.register(BCELoss)
# https://github.com/clcarwin/focal_loss_pytorch


class CrossEntropyLoss2d(torch.nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=255,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        if weight is not None:
            self.weight = torch.Tensor(weight)
        else:
            self.weight = None
        self.nll_loss = torch.nn.NLLLoss(reduction=reduction,
                                         ignore_index=ignore_index)

    def forward(self, inputs, targets, do_rmi=None):
        if self.weight is not None:
            self.nll_loss.weight = self.weight
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


mlconfig.register(CrossEntropyLoss2d)

class FocalLoss(torch.nn.Module):
    """
    alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. 
                Default: ``0.25``.
    gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. 
                Default: ``2``.
    """
    def __init__(self, gamma=0, alpha=None, size_average=True, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) 

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            # print('alpha before:', self.alpha.shape, self.alpha)
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            logpt = logpt * Variable(self.alpha[0])

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

mlconfig.register(FocalLoss)

# if __name__ == '__main__':
#     print('#### Test Case ###')
    
#     maxe = 0
#     import random
#     x = torch.rand(10, 5, 4, 4)
#     x = Variable(x.cuda())
#     l = torch.randint(5, (10, 4, 4))
#     l = Variable(l.cuda())

#     output0 = FocalLoss(gamma=0, alpha=1)(x,l)
#     output1 = torch.nn.CrossEntropyLoss()(x,l)
#     a = output0
#     b = output1
#     if abs(a-b)>maxe: maxe = abs(a-b)
#     print(a, b, maxe)


