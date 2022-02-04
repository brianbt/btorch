import math
from btorch import nn

class DepthPointWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, s1=1, s2=1, kernel_size=3, multiplier=1, **kwargs):
        """         
        -> Depthwise_Conv2d -> BN -> ReLU -> Pointwise_Conv2d -> BN -> ReLU ->
        s1: stride for deepthwise-conv
        s2: stride for pointwise-conv
        """
        if "groups" in kwargs or "stride" in kwargs:
            raise Exception("`groups` AND `stride` are reserved arguments")
        in_channels = math.ceil(in_channels*multiplier)
        out_channels = math.ceil(out_channels*multiplier)
        super(DepthPointWiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   groups=in_channels, stride=s1, padding=1,**kwargs)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=s2, **kwargs)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x=self.depthwise(x)
        x=self.batchnorm1(x)
        x=self.relu(x)
        x=self.pointwise(x)
        x=self.batchnorm2(x)
        x=self.relu(x)
        return x