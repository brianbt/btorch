import torch
from torchvision import transforms, datasets

import btorch
from btorch import nn
from btorch.vision import models

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(ResNet, self).__init__()
        self.pre_process = nn.Conv2d(1,3,1)
        self.resizer = nn.Resizer()
        self.model = models.resnet50(num_classes)
        self.last = nn.Linear(1000, 10)
    def forward(self, x):
        x=self.pre_process(x)
        x=self.resizer(x)
        x=self.model(x)
        x=torch.flatten(x,1)
        x=self.last(x)
        return x

# DataSet
transform = transforms.Compose(
    [transforms.ToTensor()])
batch_size = 256
trainset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
# Model
net = ResNet(20)

# Loss & Optimizer & Config
net._optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net._lossfn = nn.CrossEntropyLoss()
net._config = {'max_epoch':2}
net.auto_gpu()

# FIT
net.fit(trainloader)

# PREDICT
net.predict(testloader, return_combined=True).max(1)[1]
