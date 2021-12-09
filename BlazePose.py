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
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


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
            conv_input = F.pad(x, [0, 1, 0, 1], "constant", 0)  # TODO: check
            x = self.pool(x)
        else:
            conv_input = x

        conv_out = self.depth_conv(conv_input)
        conv_out = self.pointwise_conv(conv_out)

        if self.channel_pad > 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self.channel_pad], "constant", 0)

        return self.relu(conv_out + x)


class BlazePose(nn.Module):
    def __init__(self, num_keypoints: int):
        super(BlazePose, self).__init__()

        self.num_keypoints = num_keypoints

        # stem layers
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16)
            )

        # blaze blocks
        self.conv3 = BlazeBlock(16, 32, 2)
        self.conv4 = BlazeBlock(32, 64, 2)
        self.conv5 = BlazeBlock(64, 128, 2)
        self.conv6 = BlazeBlock(128, 192, 2)
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        # last several layers
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 3*self.num_keypoints, 1, 1, 0, bias=False),
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
        x += self.conv2(x)
        
        # blaze blocks
        y0 = self.conv3(x)
        y1 = self.conv4(y0)
        y2 = self.conv5(y1)
        y3 = self.conv6(y2)

        # get heat map
        x3 = self.conv7(y3)
        x2 = self.conv8(y2) + self.upsample(x3)
        x1 = self.conv9(y1) + self.upsample(x2)
        x0 = self.conv10(y0) + self.upsample(x1)
        heatmap = self.conv11(x0) # => heatmap

        # get joints
        with torch.no_grad():
            joints = x0 + y0
            joints = self.conv4(joints) + y1
            joints = self.conv5(joints) + y2
            joints = self.conv6(joints) + y3
        joints = self.conv12(joints)
        joints = self.conv13(joints)
        joints = self.conv14(joints) # => joints

        return [heatmap, joints]

if __name__ == '__main__':
    from torchsummaryX import summary

    dummy_input = torch.rand(8,3,256,256)
    model = BlazePose(33)
    profile = summary(model,dummy_input)
