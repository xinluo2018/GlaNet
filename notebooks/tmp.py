import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块 - 保持空间尺寸"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 确保中间通道数一致
        mid_channels = out_channels // 4
        
        self.conv1 = ConvBlock(in_channels, mid_channels, stride=stride)
        self.conv2 = ConvBlock(mid_channels, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity)

class TerrainGuidedAttention(nn.Module):
    """地形引导注意力模块 - 添加空间对齐"""
    def __init__(self, img_channels, dem_channels):
        super().__init__()
        # DEM特征投影
        self.dem_proj = nn.Sequential(
            nn.Conv2d(dem_channels, img_channels, 1),
            nn.BatchNorm2d(img_channels),
            nn.ReLU()
        )
        
        # 注意力生成网络
        self.attention_net = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(img_channels, img_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img_feat, dem_feat):
        # DEM特征投影
        dem_proj = self.dem_proj(dem_feat)
        
        # 确保空间尺寸一致
        if img_feat.size(2) != dem_proj.size(2) or img_feat.size(3) != dem_proj.size(3):
            # 使用双线性插值调整DEM特征尺寸
            dem_proj = F.interpolate(
                dem_proj, 
                size=img_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 生成地形引导注意力图
        concat_feat = torch.cat([img_feat, dem_proj], dim=1)
        attention = self.attention_net(concat_feat)
        
        # 注意力加权融合
        terrain_guided_feat = img_feat * attention
        
        # 残差连接
        return terrain_guided_feat + dem_proj

class GlacierSegNet(nn.Module):
    """修复后的冰川范围分割模型 - 确保空间尺寸对齐"""
    def __init__(self, num_spectral_bands=6, num_classes=1):
        super().__init__()
        
        # 多光谱影像编码器 (使用ConvNeXt-Tiny作为骨干网络)
        self.spectral_encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 修改第一层以适应6波段输入
        self.spectral_encoder.features[0][0] = nn.Conv2d(
            num_spectral_bands, 96, kernel_size=4, stride=4, padding=0
        )
        
        # DEM数据处理流
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )
        
        # 特征融合模块 (在不同阶段)
        self.fusion_stages = nn.ModuleList([
            # 第一阶段: 1/4分辨率 (地形引导注意力)
            TerrainGuidedAttention(96, 32),  # 输入:96+32, 输出:96
            
            # 第二阶段: 1/8分辨率 (地形引导注意力)
            TerrainGuidedAttention(192, 64), # 输入:192+64, 输出:192
            
            # 第三阶段: 1/16分辨率 (地形引导注意力)
            TerrainGuidedAttention(384, 128), # 输入:384+128, 输出:384
            
            # 第四阶段: 1/32分辨率 (地形引导注意力)
            TerrainGuidedAttention(768, 256)  # 输入:768+256, 输出:768
        ])
        
        # 解码器输入通道
        decoder_channels = [768, 384, 192, 96]
        
        # 特征解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(768, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(384, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, spectral_img, dem_data):
        # 处理多光谱影像
        spectral_features = []
        x = spectral_img
        
        # 获取ConvNeXt-Tiny的不同阶段特征
        for i, layer in enumerate(self.spectral_encoder.features):
            x = layer(x)
            if i in [1, 3, 5, 7]:  # 保存不同阶段的输出特征
                spectral_features.append(x)
        
        # 处理DEM数据
        dem_features = []
        d = dem_data
        
        # 逐步处理DEM并收集特征
        d = self.dem_encoder[0](d)  # Conv2d
        d = self.dem_encoder[1](d)  # BN
        d = self.dem_encoder[2](d)  # ReLU
        d = self.dem_encoder[3](d)  # MaxPool2d -> 1/4分辨率
        dem_features.append(d)
        
        d = self.dem_encoder[4](d)  # 1/8分辨率
        dem_features.append(d)
        
        d = self.dem_encoder[5](d)  # 1/16分辨率
        dem_features.append(d)
        
        d = self.dem_encoder[6](d)  # 1/32分辨率
        dem_features.append(d)
        
        # 选择正确分辨率的DEM特征
        dem_selected_features = dem_features
        
        # 特征融合 (在不同分辨率阶段)
        fused_features = []
        for i in range(4):
            fused = self.fusion_stages[i](
                spectral_features[i], 
                dem_selected_features[i]
            )
            fused_features.append(fused)
        
        # 使用最高层特征进行解码
        seg_logits = self.decoder(fused_features[3])
        
        return seg_logits


if __name__ == "__main__":
    model = GlacierSegNet(num_spectral_bands=6, num_classes=1)
    data = torch.randn(2, 6, 256, 256)  # 模拟输入 (批量大小=2)
    dem = torch.randn(2, 1, 256, 256)    # 模拟DEM输入
    output = model(data, dem)
    print(output.shape)