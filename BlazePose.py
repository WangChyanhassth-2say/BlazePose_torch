import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SwishLinear(nn.Module):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return self.linear(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            h_sigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(BlazeBlock, self).__init__()
        self.use_pooling = stride == 2
        self.channel_pad = out_channels - in_channels

        if self.use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            padding = 0
        else:
            padding = 1

        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.use_pooling:
            conv_input = F.pad(x, [0, 1, 0, 1], "constant", 0)
            x = self.pool(x)
        else:
            conv_input = x

        conv_out = self.depth_conv(conv_input)
        conv_out = self.pointwise_conv(conv_out)

        if self.channel_pad > 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self.channel_pad], "constant", 0)

        return self.relu(conv_out + x)

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = h_swish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BlazePose(nn.Module):
    def __init__(self, num_keypoints: int, return_type='two_head'):
        super(BlazePose, self).__init__()

        self.num_keypoints = num_keypoints
        self.return_type = return_type
        # stem layers
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                h_swish()
                )
        # MobileBottleneck: input:(inp, oup, k, s, exp, se, li)
        self.conv2 = MobileBottleneck(16, 16, 3, 1, 16, False, 'RE')
        self.conv2_b1 = MobileBottleneck(16, 16, 3, 1, 72, False, 'RE')
        self.conv3_b1 = MobileBottleneck(32, 32, 5, 1, 120, True, 'RE')
        self.conv3_b2 = MobileBottleneck(32, 32, 5, 1, 120, True, 'RE')
        self.conv4_b1 = MobileBottleneck(64, 64, 3, 1, 200, False, 'HS')
        self.conv4_b2 = MobileBottleneck(64, 64, 3, 1, 184, False, 'HS')
        self.conv4_b3 = MobileBottleneck(64, 64, 3, 1, 184, False, 'HS')
        self.conv5_b1 = MobileBottleneck(128, 128, 3, 1, 480, True, 'HS')
        self.conv5_b2 = MobileBottleneck(128, 128, 3, 1, 672, True, 'HS')
        self.conv6_b1 = MobileBottleneck(192, 192, 5, 1, 960, True, 'HS')
        self.conv6_b2 = MobileBottleneck(192, 192, 5, 1, 960, True, 'HS')

        # blaze blocks
        self.conv3 = BlazeBlock(16, 32, 2)
        self.conv4 = BlazeBlock(32, 64, 2)
        self.conv5 = BlazeBlock(64, 128, 2)
        self.conv6 = BlazeBlock(128, 192, 2)
        self.conv4_1 = BlazeBlock(32, 64, 2)
        self.conv5_1 = BlazeBlock(64, 128, 2)
        self.conv6_1 = BlazeBlock(128, 192, 2)
        self.conv12 = BlazeBlock(192, 192, 2)
        self.conv13 = BlazeBlock(192, 192, 2)

        # layers for the skip_connection
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 288, 1, 1, 0, bias=False),
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.Conv2d(288, 288, 3, 1, 1, groups=288, bias=False),
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.Conv2d(288, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32)
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 192, 1, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32)
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32)
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32)
            )

        # up sample layer
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        # last several layers
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, self.num_keypoints, 1, 1, 0, bias=False),
            nn.Sigmoid()
            )
        self.conv14 = nn.Sequential(
            nn.ReLU6(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish(),
            Reshape(192),
            SwishLinear(192, 3*self.num_keypoints),
            nn.Dropout(0.05),
            Reshape(3, self.num_keypoints)
            )
        
    def forward(self, x):

        # stem layers
        x = self.conv1(x)
        # dw and pw like mobilenet
        x = self.conv2(x)
        x = self.conv2_b1(x)
        # blaze blocks
        y0 = self.conv3(x)
        y0 = self.conv3_b1(y0)
        y0 = self.conv3_b2(y0)
        y1 = self.conv4(y0)
        y1 = self.conv4_b1(y1)
        y1 = self.conv4_b2(y1)
        y1 - self.conv4_b3(y1)
        y2 = self.conv5(y1)
        y2 = self.conv5_b1(y2)
        y2 = self.conv5_b2(y2)
        y3 = self.conv6(y2)
        y3 = self.conv6_b1(y3)
        y3 = self.conv6_b2(y3)

        # get heatmap
        x3 = self.conv7(y3)
        x2 = self.conv8(y2) + self.upsample2(x3)
        x1 = self.conv9(y1) + self.upsample1(x2)
        x0 = self.conv10(y0) + self.upsample0(x1)
        heatmap = self.conv11(x0) # => heatmap
        # sorry I don't use offsetmap here as it in the paper

        # get joints
        x = x0 + y0.detach()
        x = self.conv4_1(x) + y1.detach()
        x = self.conv5_1(x) + y2.detach()
        x = self.conv6_1(x) + y3.detach()
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x) # => joints
        
        if self.return_type == 'two_head':
            return x, heatmap
        elif self.return_type == 'heatmap':
            return heatmap
        else:
            return x

if __name__ == '__main__':
    from torchsummaryX import summary

    dummy_input = torch.rand(8,3,256,256)
    model = BlazePose(33, 'two_head')
    profile = summary(model,dummy_input)
