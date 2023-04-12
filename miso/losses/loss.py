import torch
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss
from allennlp.common import Registrable
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional


class LossFunctionDict(dict):
    def __init__(self,*arg,**kwargs):
        super(LossFunctionDict, self).__init__(*arg, **kwargs)
        self['MSELoss'] = MSELoss()
        self['L1Loss'] = L1Loss()
        self['MSECrossEntropyLoss'] = MSECrossEntropyLoss()
        self['BCEWithLogitsLoss'] = BCEWithLogitsLoss()

class Loss(torch.nn.Module, Registrable): 
    def __init__(self):
        super(Loss, self).__init__()
        pass

    def forward(self, output, target):
        pass 

@Loss.register("mse_cross_entropy") 
class MSECrossEntropyLoss(Loss): 
    def __init__(self):
        super(MSECrossEntropyLoss, self).__init__()
        self.mse_criterion = MSELoss()
        self.xent_criterion = BCELoss()

    def forward(self, output, target):
        mse_value = self.mse_criterion(output, target)
        
        thresholded_output = torch.gt(output, 0).float()
        thresholded_target = torch.gt(target, 0).float()

        xent_value = self.xent_criterion(thresholded_output, thresholded_target)
        if xent_value + mse_value != 0:
            harmonic_mean = 2*(xent_value * mse_value)/(xent_value + mse_value)
        else:
            # 0 by default, all 0's
            harmonic_mean = xent_value + mse_value
        return harmonic_mean

#BCEWithLogitsLoss
@Loss.register("bce_with_logits_loss")
class BCEWithLogitsLoss(Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
	
        
    
