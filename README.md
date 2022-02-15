# BTorch
[![Documentation Status](https://readthedocs.org/projects/btorch/badge/?version=latest)](https://btorch.readthedocs.io/en/latest/?badge=latest)  
BTorch is a PyTorch's useful utils library

## Requiements
PyTorch ≥ 1.10

## Install
`pip install git+https://github.com/brianbt/btorch`
## High Level Module (nn.Module)
You can use `btorch.nn` as normal `torch.nn` with more useful functions.  
You can define your model by subclassing it from `btorch.nn.Module` and everythings will be same as subclassing from `torch.nn.Module`.  
`btorch.nn.Module` provides you high-level training loop that you can define by yourself. Making the code more clean while maintain the flexibilityof PyTorch.  

The high-level methods are  
- .fit()  
- .train_net()  
- .train_epoch()  
- .test_epoch()  

Override them when necessary and train your model by just calling `.fit()`

### Usage  
```python
class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(ResNet, self).__init__()
        self.pre_process = nn.Conv2d(1,3,1)
        self.model = models.bresnet50(num_classes, include_top=False)
        self.last = nn.Linear(2048, num_classes)
    def forward(self, x):
        x=self.pre_process(x)
        x=self.model(x)
        x=torch.flatten(x, 1)
        x=self.last(x)
        return x
    # Overwrite our predict function
    @classmethod
    def predict_(cls, net, loader, device='cuda', config=None):
        net.to(device)
        net.eval()
        out = []
        with torch.inference_mode():
            for batch_idx, (inputs, _) in enumerate(loader):
                inputs =  inputs.to(device)
                logit = net(inputs)
                answer = torch.max(torch.softmax(logit,1), 1)
                out.append(answer)
        return out


net = ResNet(20)

# Loss & Optimizer & Config
net._optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net._lossfn = nn.CrossEntropyLoss()
net._config = {'max_epoch':3}

# Set GPU
device = net.auto_gpu()

# Fit model
net.fit(trainloader)

# Predict
for i in testloader:
    break
net.predict(i[0])
net.predict(testloader)
```
## Inherent Library
You can import below library and use that as pyTorch
```
from btorch import nn
import btorch.nn.functional as F
from btorch.vision import models
```