'''
author: xin luo
create: 2026.4.2
des: UNet model with timm backbone
'''

import torch
import torch.nn as nn
import timm

def conv(in_channels, out_channels, 
                kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class u3net_timm_data_ablation(nn.Module):
    def __init__(self,  backbone_name='resnet34', 
                        pretrained=True,
                        remove='rgb'):
        '''
        '''
        super(u3net_timm_data_ablation, self).__init__()
        self.decode_channels = [96, 96, 96, 96, 48]  # decoder channels for each stage
        self.up = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)  # upsample layer
        self.remove = remove
        ## encoder part
        if self.remove != 'rgb':
            self.encoder_opt = timm.create_model(backbone_name, 
                                            features_only=True, 
                                            in_chans=3, 
                                            pretrained=pretrained)
        if self.remove != 'ir':
            self.encoder_nir = timm.create_model(backbone_name, 
                                            features_only=True, 
                                            in_chans=3, 
                                            pretrained=pretrained)
        if self.remove != 'dem':
            self.encoder_dem = timm.create_model(backbone_name, 
                                            features_only=True,
                                            in_chans=1,
                                            pretrained=pretrained)
        if self.remove != 'rgb':
            self.out_channels = self.encoder_opt.feature_info.channels()  # type: ignore
        else:
            self.out_channels = self.encoder_dem.feature_info.channels()  # type: ignore
        ## decoder part (fused features)
        print("Output channels from encoder stages:", self.out_channels)
        self.DecoderBlocks = nn.ModuleList([
                conv(self.out_channels[4]*2, self.decode_channels[0]),   
                conv(self.out_channels[3]*2+self.decode_channels[0], self.decode_channels[1]),  
                conv(self.out_channels[2]*2+self.decode_channels[1], self.decode_channels[2]), 
                conv(self.out_channels[1]*2+self.decode_channels[2], self.decode_channels[3]), 
                conv(self.out_channels[0]*2+self.decode_channels[3], self.decode_channels[4])   
                ])
        self.outp = nn.Sequential(
                        nn.Conv2d(self.decode_channels[4], 1, kernel_size=3, padding=1)) 
        ## auxiliary outputs for deep supervision
        self.aux2 = nn.Conv2d(self.out_channels[2]*2, 1, 3, padding=1)
        self.aux3 = nn.Conv2d(self.out_channels[3]*2, 1, 3, padding=1) 
        self.aux4 = nn.Conv2d(self.out_channels[4]*2, 1, 3, padding=1) 

    def forward(self, x):       ## input size: 
        '''
        x: input tensor
        '''
        feas = []
        if self.remove != 'rgb':
            x_opt = x[:, :3, :, :]
            feas_opt = self.encoder_opt(x_opt)  # list of features from encoder branch 1
            feas.append(feas_opt)
        if self.remove != 'ir':
            x_nir = x[:, 3:6, :, :]
            feas_nir = self.encoder_nir(x_nir)  # list of features from encoder branch 2
            feas.append(feas_nir)
        if self.remove != 'dem':
            x_dem = x[:, 6:, :, :]
            feas_dem = self.encoder_dem(x_dem)  # list of features from encoder branch 3
            feas.append(feas_dem)
        # layer 4
        # fea_opt, fea_nir, fea_dem = feas_opt[-1], feas_nir[-1], feas_dem[-1]   #  [B, C, 16, 16]
        fea_1, fea_2 = feas[0][-1], feas[1][-1]  #  [B, C, 16, 16]
        aux4_out = self.aux4(torch.cat([fea_1, fea_2], dim=1))  # auxiliary output at layer 4
        fea_fus_4 = torch.cat([fea_1, fea_2], dim=1)  #
        fea_fus_4 = self.DecoderBlocks[0](fea_fus_4)   # fused features through decoder    
        fea_fus_4 = self.up(fea_fus_4)  # upsample to match next skip connection

        # layer 3
        # skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-2], feas_nir[-2], feas_dem[-2]         # 32x32
        skip_fea_1, skip_fea_2 = feas[0][-2], feas[1][-2]        
        aux3_out = self.aux3(torch.cat([skip_fea_1, skip_fea_2], dim=1))  # auxiliary output at layer 4
        fea_fus_3 = torch.cat([skip_fea_1, skip_fea_2], dim=1)
        fea_fus_3 = torch.cat([fea_fus_4, fea_fus_3], dim=1)  # concat skip features
        fea_fus_3 = self.DecoderBlocks[1](fea_fus_3)  # decode fused features
        fea_fus_3 = self.up(fea_fus_3)  # upsample for next stage

        # layer 2:
        # skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-3], feas_nir[-3], feas_dem[-3]    # 64x64
        skip_fea_1, skip_fea_2 = feas[0][-3], feas[1][-3]
        aux2_out = self.aux2(torch.cat([skip_fea_1, skip_fea_2], dim=1))  # auxiliary output at layer 4
        fea_fus_2 = torch.cat([skip_fea_1, skip_fea_2], dim=1)
        fea_fus_2 = torch.cat([fea_fus_3, fea_fus_2], dim=1)  # concat skip features
        fea_fus_2 = self.DecoderBlocks[2](fea_fus_2)  # decode fused features
        fea_fus_2 = self.up(fea_fus_2)  # upsample for next stage

        # layer 1:
        # skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-4], feas_nir[-4], feas_dem[-4]      ## 128x128
        skip_fea_1, skip_fea_2 = feas[0][-4], feas[1][-4]
        skip_fus_1 = torch.cat([skip_fea_1, skip_fea_2], dim=1)
        fea_fus_1 = torch.cat([fea_fus_2, skip_fus_1], dim=1)  # concat skip features
        fea_fus_1 = self.DecoderBlocks[3](fea_fus_1)  # decode fused features
        fea_fus_1 = self.up(fea_fus_1)    ## upsample for next stage

        # layer 0: 
        # skip_fea_opt, skip_fea_nir, skip_fea_dem  = feas_opt[-5], feas_nir[-5], feas_dem[-5]     ## 256x256
        skip_fea_1, skip_fea_2 = feas[0][-5], feas[1][-5]
        fea_fus_0 = torch.cat([skip_fea_1, skip_fea_2], dim=1)  # concat skip features
        fea_fus_0 = torch.cat([fea_fus_1, fea_fus_0], dim=1)  # concat skip features
        fea_fus_0 = self.DecoderBlocks[4](fea_fus_0)  # decode fused features
        fea_fus_0 = self.up(fea_fus_0)  # upsample for next stage
        logit = self.outp(fea_fus_0)  # 1x512x512
        return logit, aux2_out, aux3_out, aux4_out  # return main output and auxiliary outputs


if __name__ == '__main__':
    model = u3net_timm_data_ablation(# backbone_name='resnet34', 
                        backbone_name='efficientnet_b0',
                        pretrained=True,
                        remove='dem')
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=W=256
    out = model(x)
    print(out[0].shape)  # should be [2, 1, 256, 256] 


