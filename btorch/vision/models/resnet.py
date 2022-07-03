from collections import OrderedDict

import torch
from torch import nn
from torchvision import models


def bresnet50(classes, include_top=True, *args, **kwargs):
    model = models.resnet50(*args, **kwargs)
    if include_top is False:
        return nn.Sequential(OrderedDict([('conv1', model.conv1),
                                          ('bn1', model.bn1),
                                          ('relu', model.relu),
                                          ('maxpool', model.maxpool),
                                          ('layer1', model.layer1), ('layer2', model.layer2), ('layer3', model.layer3),
                                          ('layer4', model.layer4)]))
    model.fc = nn.Linear(2048, classes)
    return model


class ResNet_PF():
    """ ResNet Pyramid Network. It will return `out_indices` stage's feature maps.

    Args:
        version (int): number of layers in ResNet. Defaults to 50.
        pretrained (bool, optional): _description_. Defaults to False.
        progess (bool, optional): _description_. Defaults to True.
        num_classes (int, optional): _description_. Defaults to 1000.
        out_indices (tuple, optional): Output from which stages.. Defaults to (0,1,2,3,4).
    """

    def __init__(self, version=50, pretrained=False, progress=True, num_classes=1000,
                 out_indices=(0, 1, 2, 3, 4)) -> None:
        mapping = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        self.out_indices = out_indices
        self.model = mapping['version'](pretrained=pretrained, progess=progress)
        if version in [18, 34]:
            self.model.fc = nn.Linear(512, num_classes)
        elif version in [50, 101, 152]:
            self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if 0 in self.out_indices:
            out.append(x)
        x = self.model.layer1(x)
        if 1 in self.out_indices:
            out.append(x)
        x = self.model.layer2(x)
        if 2 in self.out_indices:
            out.append(x)
        x = self.model.layer3(x)
        if 3 in self.out_indices:
            out.append(x)
        x = self.model.layer4(x)
        if 4 in self.out_indices:
            out.append(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        out.append(x)
        return tuple(out)
