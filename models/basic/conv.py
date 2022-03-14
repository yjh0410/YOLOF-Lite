import torch.nn as nn


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act=True, depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True) if act else nn.Identity(),
                nn.Conv2d(c2, c2, kernel_size=1, bias=bias),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True) if act else nn.Identity()
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True) if act else nn.Identity()
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.convs(x)
