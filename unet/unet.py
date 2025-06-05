import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k//2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        filters = [48, 64, 128, 256, 512]

        self.b0 = ConvBlock(3, filters[0], 3)
        self.b1 = ConvBlock(filters[0], filters[0], 3)

        self.b2 = ConvBlock(filters[0], filters[1], 3)
        self.b3 = ConvBlock(filters[1], filters[1], 3)

        self.b4 = ConvBlock(filters[1], filters[2], 3)
        self.b5 = ConvBlock(filters[2], filters[2], 3)

        self.b6 = ConvBlock(filters[2], filters[3], 3)
        self.b7 = ConvBlock(filters[3], filters[3], 3)

        self.b8 = ConvBlock(filters[3], filters[4], 3)
        self.b9 = ConvBlock(filters[4], filters[4], 3)

        self.maxpool = nn.MaxPool2d(3, 2)

        self.h0 = ConvBlock(filters[4] + filters[3], filters[3], 1)
        self.h1 = ConvBlock(filters[3], filters[3], 3)

        self.h2 = ConvBlock(filters[3] + filters[2], filters[2], 1)
        self.h3 = ConvBlock(filters[2], filters[2], 3)

        self.h4 = ConvBlock(filters[2] + filters[1], filters[1], 1)
        self.h5 = ConvBlock(filters[1], filters[1], 3)

        self.h6 = ConvBlock(filters[1] + filters[0], filters[0], 1)
        self.h7 = ConvBlock(filters[0], filters[0], 3)

        self.h8 = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)

        b2 = self.b2(self.maxpool(b1))
        b3 = self.b3(b2) # /2

        b4 = self.b4(self.maxpool(b3))
        b5 = self.b5(b4) # /4

        b6 = self.b6(self.maxpool(b5))
        b7 = self.b7(b6) # /8

        b8 = self.b8(self.maxpool(b7))
        b9 = self.b9(b8) # /16

        u9 = F.interpolate(b9, size=b7.shape[-2:], mode='bilinear', align_corners=True)
        h0 = self.h0(torch.concat([u9, b7], dim=1))
        h1 = self.h1(h0)

        u1 = F.interpolate(h1, size=b5.shape[-2:], mode='bilinear', align_corners=True)
        h2 = self.h2(torch.concat([u1, b5], dim=1))
        h3 = self.h3(h2)

        u3 = F.interpolate(h3, size=b3.shape[-2:], mode='bilinear', align_corners=True)
        h4 = self.h4(torch.concat([u3, b3], dim=1))
        h5 = self.h5(h4)

        u5 = F.interpolate(h5, size=b1.shape[-2:], mode='bilinear', align_corners=True)
        h6 = self.h6(torch.concat([u5, b1], dim=1))
        h7 = self.h7(h6)

        return self.h8(h7)