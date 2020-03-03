import math
from collections import OrderedDict
import torch
from torch import nn
import torch.utils.model_zoo


HASH = '1d3f7974'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_V(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_V(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = OrderedDict([  # original model layers in sequence
            ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)),
                ('norm1', nn.BatchNorm2d(64)),
                ('nonlin1', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)),
                ('norm2', nn.BatchNorm2d(64)),
                ('nonlin2', nn.ReLU(inplace=True)),
                ('output', Identity())
            ]))),
            ('V2', CORblock_V(64, 128, times=2)),
            ('V4', CORblock_V(128, 256, times=4)),
            ('IT', CORblock_V(256, 512, times=2)),
            ('decoder', nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(512, 1000)),
                ('output', Identity())
            ])))
        ])

        # add modules to be present in state dict
        for key, module in self.m.items():
            self.add_module(key, module)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inp, stims):
        x = self.m['V1'](inp) + stims['V1']
        x = self.m['V2'](x) + stims['V2']
        x = self.m['V4'](x) + stims['V4']
        x = self.m['IT'](x) + stims['IT']
        return self.m['decoder'](x)


def cornet_v(pretrained=False, map_location=None, **kwargs):
    model_hash = '1d3f7974'
    model = CORnet_V(**kwargs)
    model = torch.nn.DataParallel(model)
    if pretrained:
        url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data['state_dict'])
    return model


if __name__ == '__main__':
    model = cornet_v(pretrained=True)
