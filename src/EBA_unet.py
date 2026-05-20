import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# 双卷积模块（DoubleConv）
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# 下采样模块（Down）
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


# 上采样模块（Up）
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 输出卷积层（OutConv）
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


# CBAM 注意力模块（CBAMBlock）
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力 - 修正输入通道数为2
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 修正输入通道数为2（avg_pool和max_pool拼接后的通道数）
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        return x


# Sobel 边缘检测函数
def sobel_edge_detection(input_tensor):
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]])
    # 获取输入张量的通道数
    num_channels = input_tensor.size(1)
    # 调整 Sobel 滤波器的通道数并确保与输入设备一致
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1).to(input_tensor.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1).to(input_tensor.device)
    # 确保输入是 4D (B, C, H, W)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # 如果是 3D，添加批次维度
    grad_x = F.conv2d(input_tensor, sobel_x, padding=1, groups=num_channels)
    grad_y = F.conv2d(input_tensor, sobel_y, padding=1, groups=num_channels)
    sobel_edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    sobel_edges = sobel_edges / (sobel_edges.max() + 1e-8)
    return sobel_edges


# U-Net 类
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, bilinear=True, base_c=64):
        super(UNet, self).__init__()
        # 输入卷积层
        self.in_conv = DoubleConv(in_channels, base_c)
        # 下采样部分
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        # 上采样部分 - 修正in_channels为拼接后的通道数
        self.up1 = Up(base_c * 16 + base_c * 8, base_c * 8)  # 512 + 256 = 768
        self.up2 = Up(base_c * 8 + base_c * 4, base_c * 4)   # 256 + 128 = 384
        self.up3 = Up(base_c * 4 + base_c * 2, base_c * 2)   # 128 + 64 = 192
        self.up4 = Up(base_c * 2 + base_c, base_c)         # 64 + 32 = 96
        # 输出层
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits}


# 结合 Sobel 和 CBAM 的 U-Net 扩展类
class EBAUNet(UNet):
    def __init__(self, in_channels=1, num_classes=2, bilinear=True, base_c=32):
        # 使用原始的in_channels初始化父类，避免影响后续网络层的通道数
        super(EBAUNet, self).__init__(in_channels=in_channels,
                                      num_classes=num_classes,
                                      bilinear=bilinear,
                                      base_c=base_c)
        # 修改输入卷积层，使其能够处理拼接后的通道数
        # 原始输入通道数(in_channels) + Sobel边缘图通道数(in_channels)
        self.in_conv = DoubleConv(in_channels * 2, base_c)
        # CBAM 模块
        self.cbam1 = CBAMBlock(base_c)
        self.cbam2 = CBAMBlock(base_c * 2)
        self.cbam3 = CBAMBlock(base_c * 4)
        self.cbam4 = CBAMBlock(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Sobel 边缘检测
        sobel_edges = sobel_edge_detection(x)
        # 将 Sobel 边缘图和输入图像拼接
        x_with_edges = torch.cat([x, sobel_edges], dim=1)
        # 网络的前向传播
        x1 = self.in_conv(x_with_edges)
        x1 = self.cbam1(x1)  # 使用 CBAM
        x2 = self.down1(x1)
        x2 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits,
                "edge_map": sobel_edges }