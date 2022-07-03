import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, n=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        return out


class Resizer(nn.Module):
    """learnable resizer https://arxiv.org/pdf/2103.09950.pdf

      Attributes:
          scale_factor (float): multiplier for spatial size. If scale_factor is a tuple, its length has to match input.dim().
          r (int): number of Residual Blocks
          n (int): number of mapping channel
      """

    def __init__(self, scale_factor=0.5, r=1, n=16, interpolate='bilinear'):
        super(Resizer, self).__init__()
        self.interpolate = interpolate
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=7, stride=1, padding=3)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(n, n, kernel_size=1, stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n)

        self.resblock = []
        self._resblock = OrderedDict()
        for i in range(r):
            self._resblock[f'resblock{i}'] = ResBlock(n)
        self.resblock = nn.Sequential(self._resblock)

        self.conv3 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n)

        self.conv4 = nn.Conv2d(n, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        main = self.conv1(x)
        main = self.leakyrelu1(main)
        main = self.conv2(main)
        main = self.leakyrelu2(main)
        main = self.bn1(main)
        main_path = F.interpolate(main, scale_factor=self.scale_factor, mode=self.interpolate)
        main = self.resblock(main_path)
        main = self.conv3(main)
        main = self.bn2(main)
        main += main_path
        main = self.conv4(main)
        main += F.interpolate(x, scale_factor=self.scale_factor, mode=self.interpolate)
        return main


class PositionalEncoding(nn.Module):
    """ PositionalEncoding Module

      https://pytorch.org/tutorials/beginner/transformer_tutorial.html
      Expected input size: (batch, seq_len, d_model) if batch_first, otherwise (seq_len, batch, d_model)

      Args:
          d_model (int): embedding dimension
          max_len (int): maximum sequence length (sqe_len)

      Shape:
          Input: (batch, seq_len, d_model) if batch_first, otherwise (seq_len, batch, d_model)
          Output: (batch, seq_len, d_model) if batch_first, otherwise (seq_len, batch, d_model)
      """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            Args:
                x: Tensor, shape [seq_len, batch_size, embedding_dim]
            """
        if self.batch_first:
            x = x.transpose(1, 0)
        x = x + self.pe[:x.size(0)]
        if self.batch_first:
            x = x.transpose(1, 0)
        return self.dropout(x)
