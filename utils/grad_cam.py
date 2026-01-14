## author: xin luo
## create: 2026.1.7
## des: Grad-CAM for 2-class U-Net model

import torch
import torch.nn.functional as F


# Grad-CAM 实现类
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Grad-CAM for U-Net model 
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = torch.zeros(1)
        self.activations = torch.zeros(1)
        
        # register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        """Save forward activations"""
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        """Save backward gradients"""
        self.gradients = grad_output[0]
    
    def forward(self, x):
        """
        Calculate Grad-CAM heatmap
        
        params:
        x: input image tensor [1, C, H, W]
        
        returns:
        heatmap: Grad-CAM heatmap
        pred_mask: predicted segmentation mask
        """
        # Forward pass
        pred_mask = self.model(x)        
        target = pred_mask.clone()
        # Zero gradients
        self.model.zero_grad()        
        pred_mask.backward(gradient=target, retain_graph=True)
        # Calculate gradient weights
        pooled_gradients = torch.mean(self.gradients, dim=(2, 3))     
        # Weighted activations
        weighted_activations = torch.zeros_like(self.activations)
        for i in range(pooled_gradients.shape[0]):
            weighted_activations[:, i, :, :] = pooled_gradients[i] * self.activations[:, i, :, :]

        # Compute heatmap
        heatmap = torch.sum(weighted_activations, dim=1).squeeze()
        
        # Apply ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)        
        return heatmap.detach().cpu().numpy(), pred_mask.detach().cpu().numpy().squeeze()
