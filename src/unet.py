from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):#卷积层
    def __init__(self, in_channels, out_channels, mid_channels=None):#三通道
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),#二维卷积层，输入通道数为 in_channels，输出通道数为 mid_channels，卷积核大小为 3x3，填充为1，不使用偏置
            nn.BatchNorm2d(mid_channels),#二维批量归一化层，用于加速模型训练和提高模型的稳定性
            nn.ReLU(inplace=True),#激活函数，inplace=True 表示直接在原张量上进行修改，节省内存
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)#同上，只是改变了输出通道
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):#接收输入通道数 in_channels 和输出通道数 out_channels 作为参数
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),#池化层，池化核大小为2x2，步长为2，用于对输入特征图进行下采样。
            DoubleConv(in_channels, out_channels)#再次卷积
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):#bilinear-双线性插值
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#使用双线性插值对输入特征图进行上采样，缩放因子为 2
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)#卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)#使用转置卷积对输入特征图进行上采样
            self.conv = DoubleConv(in_channels, out_channels)#卷积

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:#定义前向传播方法，接收两个输入张量x1和x2
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)#对输入就行卷积操作
        x2 = self.down1(x1)#开始下采样
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)#进行上采样操作，并与相应的下采样层的特征图进行拼接
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}