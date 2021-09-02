import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x1': lambda C, stride, affine: nn.AvgPool2d((3, 1), stride=(stride, 1), padding=(1, 0),
                                                           count_include_pad=False),
    'max_pool_3x1': lambda C, stride, affine: nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_3x1': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'conv_5x1': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'conv_7x1': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'conv_9x1': lambda C, stride, affine: SepConv(C, C, 9, stride, 4, affine=affine),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(C_in, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, :].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, (1, 1), stride=(2, 1), padding=(0, 0), bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, (1, 1), stride=(2, 1), padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, :])], dim=1)
        out = self.bn(out)
        return out
