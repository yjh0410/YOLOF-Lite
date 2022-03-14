import torch
import torchvision.ops
from torch import nn
import math


class DyConv(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 kernel_size=3,
                 stride=1,
                 padding=1):

        super(DyConv, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_dims))
        
        # init        
        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_dims * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def forward(self, x, offset, mask):
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x
        