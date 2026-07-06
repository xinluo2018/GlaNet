'''
author: xin luo
create: 2026.4.2
des: U-Net model with timm backbone
'''

import torch
import torch.nn as nn
import timm

def conv(in_channels, out_channels, 
                kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        # nn.GroupNorm(1, out_channels),  # group normalization  
        # GlobalBatchNorm2d(out_channels),  # global batch normalization      
        nn.ReLU(inplace=True)
        )

class u2net_timm(nn.Module):
    def __init__(self, backbone_name='resnet34', 
                        pretrained=True):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_timm, self).__init__()
        # self.decode_channels = [64, 64, 64, 64, 32]  # decoder channels for each stage
        self.decode_channels = [128, 128, 128, 128, 64]  # decoder channels for each stage
        self.up = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)  # upsample layer

        ## encoder part
        self.encoder_opt = timm.create_model(backbone_name, 
                                        features_only=True, 
                                        in_chans=6, 
                                        pretrained=pretrained)
        self.encoder_dem = timm.create_model(backbone_name, 
                                        features_only=True,
                                        in_chans=1,
                                        pretrained=pretrained)
        self.out_channels = self.encoder_opt.feature_info.channels()
        ## decoder part (fused features)
        self.DecoderBlocks = nn.ModuleList([
                conv(self.out_channels[-1]*2, self.decode_channels[0]),   
                conv(self.out_channels[-2]*2+self.decode_channels[0], self.decode_channels[1]),  
                conv(self.out_channels[-3]*2+self.decode_channels[1], self.decode_channels[2]), 
                conv(self.out_channels[-4]*2+self.decode_channels[2], self.decode_channels[3]), 
                conv(self.out_channels[-5]*2+self.decode_channels[3], self.decode_channels[4])   
                ])
        self.outp = nn.Sequential(
                        nn.Conv2d(self.decode_channels[4], 1, kernel_size=3, padding=1)) 


    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_opt, x_dem = x[:, :6, :, :], x[:, 6:, :, :]
        ## encoder part
        feas_opt = self.encoder_opt(x_opt)  # list of features from encoder branch 1
        feas_dem = self.encoder_dem(x_dem)  # list of features from encoder branch 2
        # layer 4
        fea_opt, fea_dem = feas_opt[-1], feas_dem[-1]   #  [B, C, 16, 16]
        fea_fus_4 = torch.cat([fea_opt, fea_dem], dim=1)  #   
        fea_fus_4 = self.DecoderBlocks[0](fea_fus_4)   # fused features through decoder    
        fea_fus_4 = self.up(fea_fus_4)  # upsample to match next skip connection
        # layer 3
        skip_fea_opt, skip_fea_dem = feas_opt[-2], feas_dem[-2]        
        fea_fus_3 = torch.cat([fea_fus_4, skip_fea_opt, skip_fea_dem], dim=1)  # concat skip features
        fea_fus_3 = self.DecoderBlocks[1](fea_fus_3)  # decode fused features
        fea_fus_3 = self.up(fea_fus_3)  # upsample for next stage
        # layer 2:
        skip_fea_opt, skip_fea_dem = feas_opt[-3], feas_dem[-3]
        fea_fus_2 = torch.cat([fea_fus_3, skip_fea_opt, skip_fea_dem], dim=1)  # concat skip features
        fea_fus_2 = self.DecoderBlocks[2](fea_fus_2)  # decode fused features
        fea_fus_2 = self.up(fea_fus_2)  # upsample for next stage
        # layer 1:
        skip_fea_opt, skip_fea_dem = feas_opt[-4], feas_dem[-4]
        fea_fus_1 = torch.cat([fea_fus_2, skip_fea_opt, skip_fea_dem], dim=1)  # concat skip features
        fea_fus_1 = self.DecoderBlocks[3](fea_fus_1)  # decode fused features
        fea_fus_1 = self.up(fea_fus_1)    ## upsample for next stage

        # layer 0: 
        skip_fea_opt, skip_fea_dem = feas_opt[-5], feas_dem[-5]
        fea_fus_0 = torch.cat([fea_fus_1, skip_fea_opt, skip_fea_dem], dim=1)  # concat skip features
        fea_fus_0 = self.DecoderBlocks[4](fea_fus_0)  # decode fused features
        fea_fus_0 = self.up(fea_fus_0)  # upsample for next stage
        logit = self.outp(fea_fus_0)  # 1x512x512
        return logit


if __name__ == '__main__':
    model = u2net_timm(# backbone_name='resnet34', 
                        backbone_name='efficientnet_b0',
                        pretrained=True)
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=W=256
    out = model(x)
    print(out.shape)  # should be [2, 1, 256, 256] 

