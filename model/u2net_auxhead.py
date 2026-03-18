'''
author: xin luo
create: 2025.12.16
des: a dual branch U-Net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
try:    
    from .attention import CBAMBlock
except ImportError:
    from attention import CBAMBlock

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class AuxHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor,
                target_size: tuple) -> torch.Tensor:
        x = F.interpolate(x, size=target_size,
                          mode="bilinear", align_corners=False)
        return self.head(x)


class u2net_(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.down_conv1_b1 = conv3x3_bn_relu(self.num_bands_b1, 16)  
        self.down_conv2_b1 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b1 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b1 = conv3x3_bn_relu(64, 128)
        ### branch 2
        self.down_conv1_b2 = conv3x3_bn_relu(self.num_bands_b2, 16)
        self.down_conv2_b2 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b2 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b2 = conv3x3_bn_relu(64, 128)
        ## decoder part (fused features)
        self.up_conv1 = conv3x3_bn_relu(384, 64)  # in: 128+128+64+64 = 384
        self.cbam1 = CBAMBlock(channel=64, reduction=4, kernel_size=7)  # CBAM attention after 1st block
        self.up_conv2 = conv3x3_bn_relu(128, 64)   # in: 64+32+32
        self.cbam2 = CBAMBlock(channel=64, reduction=4, kernel_size=7)  # CBAM attention after 2nd block
        self.up_conv3 = conv3x3_bn_relu(96, 32)   # in: 64+16+16
        self.cbam3 = CBAMBlock(channel=32, reduction=4, kernel_size=7)  # CBAM attention after 3rd block


        # #  深监督辅助头（训练时用，推理时丢弃）
        self.aux1 = AuxHead(64)  # 输入为拼接后的 x3_decoder: 64
        self.aux2 = AuxHead(64)   # 输入为拼接后的 x2_decoder: 64
        self.aux3 = AuxHead(32)   # decoder level3 输出，1/2
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
                        ) 

    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        H0, W0 = x.shape[-2:]
        ## encoder part
        ### branch 1 (scene image)
        x1_b1 = self.down_conv1_b1(x_b1)                 
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: 1/2
        x2_b1 = self.down_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.down_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.down_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        ### branch 2 (DEM)
        x1_b2 = self.down_conv1_b2(x_b2)              
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: 1/2
        x2_b2 = self.down_conv2_b2(x1_b2)              
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x3_b2 = self.down_conv3_b2(x2_b2)              
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x4_b2 = self.down_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16
        ## decoder part
        x4_up = torch.cat([self.up(x4_b1), self.up(x4_b2), x3_b1, x3_b2], dim=1)  # 128+128+64+64 = 384
        x3_decoder = self.up_conv1(x4_up)  
        x3_up = torch.cat([self.up(x3_decoder), x2_b1, x2_b2], dim=1)   # 64+32+32
        x2_decoder = self.up_conv2(x3_up)       
        x2_up = torch.cat([self.up(x2_decoder), x1_b1, x1_b2], dim=1)   # 64+16+16
        x1_decoder = self.up_conv3(x2_up)        #  32
        x1_up = self.up(x1_decoder)              #  
        prob = self.outp(x1_up)                  #  1
        if self.training:
            aux1 = self.aux1(x3_decoder, (H0, W0))
            aux2 = self.aux2(x2_decoder, (H0, W0))
            aux3 = self.aux3(x1_decoder, (H0, W0))
            return prob, aux1, aux2, aux3
        return prob          
    
if __name__ == "__main__":
    model = u2net_(num_bands_b1=3, num_bands_b2=4)
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=256, W=256
    model.train()
    prob, aux1, aux2, aux3 = model(x)
    print("Output shapes (training mode):")
    print("Prob:", prob.shape)      # (2, 1, 256, 256)
    print("Aux1:", aux1.shape)      # (2, 1, 256, 256)
    print("Aux2:", aux2.shape)      # (2, 1, 256, 256)
    print("Aux3:", aux3.shape)      # (2, 1, 256, 256)
    model.eval()
    with torch.no_grad():
        prob = model(x)
    print("\nOutput shape (inference mode):")
    print("Prob:", prob.shape)      # (2, 1, 256, 256)