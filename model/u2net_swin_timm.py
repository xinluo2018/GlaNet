'''
author: xin luo
create: 2026.4.2
des: Dual-branch U-Net model with Swin Transformer V2 backbone
'''

import torch
import torch.nn as nn
import timm

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class u2net_swin_timm(nn.Module):
    def __init__(self, num_bands_b1=6, 
                       num_bands_b2=1, 
                       img_size=512,
                       backbone_name='swinv2_base_window8_256', 
                       pretrained=True):
        super().__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        
        self.decode_channels = [512, 256, 128, 64] 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        # ==========================================
        # 1. Encoder
        self.encoder_opt = timm.create_model(backbone_name, 
                                        features_only=True, 
                                        img_size=img_size,
                                        in_chans=num_bands_b1, 
                                        pretrained=pretrained)
        self.encoder_dem = timm.create_model(backbone_name, 
                                        features_only=True,
                                        img_size=img_size,
                                        in_chans=num_bands_b2,
                                        pretrained=pretrained)

        # out_channels of each stage 
        self.out_channels = self.encoder_opt.feature_info.channels()
        
        # ==========================================
        # 2. Decoder 部分 (针对 Swin 4个Stage 的通道拼装计算)
        # ==========================================
        self.DecoderBlocks = nn.ModuleList([
            conv3x3_bn_relu(self.out_channels[3]*2, self.decode_channels[0]),   
            conv3x3_bn_relu(self.decode_channels[0] + self.out_channels[2]*2, self.decode_channels[1]),  
            conv3x3_bn_relu(self.decode_channels[1] + self.out_channels[1]*2, self.decode_channels[2]), 
            conv3x3_bn_relu(self.decode_channels[2] + self.out_channels[0]*2, self.decode_channels[3])   
        ])
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            conv3x3_bn_relu(self.decode_channels[3], 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        ) 

    def forward(self, x):       
        '''
        x: input tensor, size: (B, num_bands_b1+num_bands_b2, H, W)
        '''
        # 分离光学与 DEM 数据
        x_opt = x[:, :self.num_bands_b1, :, :]
        x_dem = x[:, self.num_bands_b1:, :, :]
        # Encoder 
        feas_opt = self.encoder_opt(x_opt)  
        feas_dem = self.encoder_dem(x_dem)  
        feas_opt = [ feat.permute(0, 3, 1, 2) for feat in feas_opt if feat is not None]
        feas_dem = [ feat.permute(0, 3, 1, 2) for feat in feas_dem if feat is not None]
        # 提取最深层特征并融合 (Bottleneck)
        fea_fus = torch.cat([feas_opt[-1], feas_dem[-1]], dim=1)  
        fea_fus = self.DecoderBlocks[0](fea_fus)      
        fea_fus = self.up(fea_fus)  
        
        # Skip connections (反转前面的特征图列表，不包含最后一层)
        skips_fea_opt = list(reversed(feas_opt[:-1]))
        skips_fea_dem = list(reversed(feas_dem[:-1]))
        
        # 此时 skips_fea_opt 长度为 3
        for i in range(len(skips_fea_opt)):
            skip_fea_opt = skips_fea_opt[i]
            skip_fea_dem = skips_fea_dem[i]            
            fea_fus = torch.cat([fea_fus, skip_fea_opt, skip_fea_dem], dim=1)  
            fea_fus = self.DecoderBlocks[i+1](fea_fus)  
            if i < len(skips_fea_opt) - 1:
                fea_fus = self.up(fea_fus)  

        logit = self.final_up(fea_fus)  
        return logit

if __name__ == '__main__':
    model = u2net_swin_timm(num_bands_b1=6, 
                         num_bands_b2=1,
                         img_size = 512, 
                         backbone_name='swinv2_base_window8_256',
                         pretrained=True)
                         
    x = torch.randn(2, 7, 512, 512)  
    out = model(x)
    print(out.shape)  # [2, 1, 256, 256]