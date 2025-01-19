import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TwoConvBlock, self).__init__()
        # todo
        # initialize the block
        self.conv2d_relu_stack = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        # todo
        # implement the forward path
        x = self.conv2d_relu_stack(x)
        return x


class DownStep(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DownStep, self).__init__()
        # todo
        # initialize the down path
        self.conv2d = TwoConvBlock(input_channel, output_channel)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # todo
        # implement the forward path
        x = self.conv2d(x)
        pooled_x = self.max_pooling(x)
        return x, pooled_x

class UpStep(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UpStep, self).__init__()
        # todo
        # initialize the up path
        self.upconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_block = TwoConvBlock(input_channel, output_channel)

    def forward(self, up_conv_x, copy_and_crop_x):
        # todo
        # implement the forward path
        up_conv_x = self.upconv(up_conv_x)
        # concat channels of x1 and x2
        diff_y = copy_and_crop_x.size()[2] - up_conv_x.size()[2]
        diff_x = copy_and_crop_x.size()[3] - up_conv_x.size()[3]
        copy_and_crop_x = copy_and_crop_x[:, :, diff_y // 2: -diff_y // 2, diff_x // 2: -diff_x // 2]
        x = torch.cat([copy_and_crop_x, up_conv_x], dim=1)

        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # todo
        # initialize the complete model
        # contracting path
        self.down1 = DownStep(1, 64)
        self.down2 = DownStep(64, 128)
        self.down3 = DownStep(128, 256)
        self.down4 = DownStep(256, 512)

        # bottom layer
        self.bottom = TwoConvBlock(512, 1024)

        # expansive path
        self.up1 = UpStep(1024, 512)
        self.up2 = UpStep(512, 256)
        self.up3 = UpStep(256, 128)
        self.up4 = UpStep(128, 64)

        # last layer
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # todo
        # implement the forward path
        # contracting path
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        # bottom layer
        x = self.bottom(x)

        # expansive path
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # output - [B, C, W, H]
        x = self.final_conv(x)

        # apply activation function (sigmoid)
        x = torch.sigmoid(x)
        return x
