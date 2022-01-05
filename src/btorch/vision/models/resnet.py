import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from collections import OrderedDict

def bresnet50(classes, include_top=True, *args, **kwargs):
    model = models.resnet50(*args, **kwargs)
    model.fc = nn.Linear(2048, classes)
    if include_top is False:
        return nn.Sequential(OrderedDict([('conv1', model.conv1), 
                                          ('bn1',model.bn1), 
                                          ('relu',model.relu), 
                                          ('maxpool',model.maxpool),
                                          ('layer1',model.layer1), ('layer2', model.layer2), ('layer3',model.layer3), ('layer4',model.layer4)]))
    return model


