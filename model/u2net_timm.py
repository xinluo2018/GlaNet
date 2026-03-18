import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class u2net_timm(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2, backbone_name='resnet34'):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_timm, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.decode_channels = [64, 64, 64, 64, 32]  # decoder channels for each stage
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        self.encoder_opt = timm.create_model(backbone_name, 
                                            features_only=True, 
                                            in_chans=num_bands_b1, 
                                            out_indices=(0, 1, 2, 3, 4),
                                            # norm_layer=nn.Identity
                                            )
        self.encoder_dem = timm.create_model(backbone_name, 
                                            features_only=True,
                                            in_chans=num_bands_b2,
                                            out_indices=(0, 1, 2, 3, 4),
                                            # norm_layer=nn.Identity
                                            )
        self.out_channels = self.encoder_opt.feature_info.channels()
        ## decoder part (fused features)
        self.DecoderBlocks = nn.ModuleList([
                conv3x3_bn_relu(self.out_channels[-1]*2, self.decode_channels[0]),   
                conv3x3_bn_relu(self.out_channels[-2]*2+self.decode_channels[0], self.decode_channels[1]),  
                conv3x3_bn_relu(self.out_channels[-3]*2+self.decode_channels[1], self.decode_channels[2]), 
                conv3x3_bn_relu(self.out_channels[-4]*2+self.decode_channels[2], self.decode_channels[3]), 
                conv3x3_bn_relu(self.out_channels[-5]*2+self.decode_channels[3], self.decode_channels[4])   
                ])
        self.outp = nn.Sequential(
                        nn.Conv2d(self.decode_channels[4], 1, kernel_size=3, padding=1),
                        ) 
    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_opt, x_dem = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        ## encoder part
        feas_opt = self.encoder_opt(x_opt)  # list of features from encoder branch 1
        feas_dem = self.encoder_dem(x_dem)  # list of features from encoder branch 2
        fea_opt, fea_dem = feas_opt[-1], feas_dem[-1]   #   
        fea_fus = torch.cat([fea_opt, fea_dem], dim=1)  #   
        fea_fus = self.DecoderBlocks[0](fea_fus)   # fused features through decoder    
        fea_fus = self.up(fea_fus)  # upsample to match next skip connection
        # skip connections: 
        skips_fea_opt = list(reversed(feas_opt[:-1]))
        skips_fea_dem = list(reversed(feas_dem[:-1]))
        for i, skips_fea_opt in enumerate(skips_fea_opt):
            skip_fea_dem = skips_fea_dem[i]
            fea_fus = torch.cat([fea_fus, skips_fea_opt, skip_fea_dem], dim=1)  # concat skip features
            fea_fus = self.DecoderBlocks[i+1](fea_fus)  # decode fused features
            fea_fus = self.up(fea_fus)  # upsample for next stage
        logit = self.outp(fea_fus)  # 1x256x256
        return logit

if __name__ == '__main__':
    model = u2net_timm(num_bands_b1=6, num_bands_b2=1, backbone_name='resnet34')
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=W=256
    out = model(x)
    print(out.shape)  # should be [2, 1, 256, 256] 



