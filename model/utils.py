import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientConv2d(nn.Module):
    def __init__(self, in_channels=1, mode='sobel'):
        """
        梯度卷积层
        - mode: 'sobel', 'scharr', 'prewitt', 'roberts'
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'sobel':
            kernel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32)
            kernel_y = kernel_x.t()
        elif mode == 'scharr':
            kernel_x = torch.tensor([[-3, 0, 3],
                                     [-10, 0, 10],
                                     [-3, 0, 3]], dtype=torch.float32)
            kernel_y = kernel_x.t()
        elif mode == 'prewitt':
            kernel_x = torch.tensor([[-1, 0, 1],
                                     [-1, 0, 1],
                                     [-1, 0, 1]], dtype=torch.float32)
            kernel_y = kernel_x.t()
        elif mode == 'roberts':
            kernel_x = torch.tensor([[1, 0],
                                     [0, -1]], dtype=torch.float32)
            kernel_y = torch.tensor([[0, 1],
                                     [-1, 0]], dtype=torch.float32)
        
        # 扩展为多通道卷积核
        kernel_x = kernel_x.expand(in_channels, 1, *kernel_x.shape)
        kernel_y = kernel_y.expand(in_channels, 1, *kernel_y.shape)
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
    
    def forward(self, x):
        # 计算x和y方向的梯度
        padding = self.kernel_x.shape[-1] // 2
        grad_x = F.conv2d(x, self.kernel_x, padding=padding, groups=x.shape[1])
        grad_y = F.conv2d(x, self.kernel_y, padding=padding, groups=x.shape[1])
        
        # 计算梯度幅值和方向
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        direction = torch.atan2(grad_y, grad_x)
        
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
        direction = (direction - direction.min()) / (direction.max() - direction.min() + 1e-6)
        grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-6)
        grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min() + 1e-6)
        tensor_combine = torch.cat([x, magnitude, direction, grad_x, grad_y], dim=1) 
        return tensor_combine

if __name__ == "__main__":
    # 测试代码
    input_tensor = torch.randn(4, 1, 512, 512)  # batch_size=4, channels=1, height=512, width=512
    grad_conv = GradientConv2d(in_channels=1, mode='sobel')
    tensor_combine = grad_conv(input_tensor)
    print("Combined tensor shape:", tensor_combine.shape)  # 应该是 (4, 5, 512, 512)
