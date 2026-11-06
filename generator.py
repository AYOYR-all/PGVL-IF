import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models import VGG16_Weights
from .unet_blocks import IREncoder, UNetDecoder

class VisibleStyleEncoder(nn.Module): 
    def __init__(self, vis_in_ch=3, base_ch=64):
        super().__init__()
         
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
         
        self.layer1 = vgg[:4]     
        self.layer2 = vgg[:9]    
        self.layer3 = vgg[:16]    
        self.layer4 = vgg[:23]   
         
        for param in self.parameters():
            param.requires_grad = False
         
        self.need_adapt = (base_ch != 64)
        if self.need_adapt:
            self.adapt1 = nn.Conv2d(64, base_ch, 1)
            self.adapt2 = nn.Conv2d(128, base_ch*2, 1)
            self.adapt3 = nn.Conv2d(256, base_ch*4, 1)
            self.adapt4 = nn.Conv2d(512, base_ch*8, 1)
    
    def forward(self, x): 
        
        with torch.no_grad():
            s1 = self.layer1(x)
            s2 = self.layer2(x)
            s3 = self.layer3(x)
            s4 = self.layer4(x)
         
        if self.need_adapt:
            s1 = self.adapt1(s1)
            s2 = self.adapt2(s2)
            s3 = self.adapt3(s3)
            s4 = self.adapt4(s4)
        
        return [s1, s2, s3, s4]

class styleSPACE(nn.Module):
    def __init__(self, norm_channels, style_channels, alpha_init=0.5):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False) 
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(style_channels, 256, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_channels, 3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_channels, 3, padding=1)
         
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
    def forward(self, x, style_map):
        
        B, C, H, W = x.shape
        
        if style_map.shape[2:] != (H, W):
            style_map = F.interpolate(
                style_map, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        normalized = self.norm(x) 
        
        style_feat = self.mlp_shared(style_map)
        gamma = self.mlp_gamma(style_feat)
        beta = self.mlp_beta(style_feat)
         
        alpha = torch.sigmoid(self.alpha)   
        gamma = (1 - alpha) + alpha * gamma   
        beta = alpha * beta
        
        return gamma * normalized + beta

class skipcon(nn.Module):
    def __init__(self, channels):
        super().__init__() 
        # 高频提取
        self.register_buffer('laplacian_kernel', 
            torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                        dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        )
 
        self.high_freq_weight = nn.Parameter(torch.tensor(0.9))  # 高频保留红外
        self.low_freq_weight = nn.Parameter(torch.tensor(0.7))   # 低频保留红外
        
    def forward(self, ir_feat, styled_feat):
        C = ir_feat.shape[1]
         
        kernel = self.laplacian_kernel.repeat(C, 1, 1, 1) 
        
        ir_high = F.conv2d(ir_feat, kernel, padding=1, groups=C)
        styled_high = F.conv2d(styled_feat, kernel, padding=1, groups=C)
         
        ir_low = ir_feat - ir_high
        styled_low = styled_feat - styled_high
         
        w_high = torch.sigmoid(self.high_freq_weight)
        w_low = torch.sigmoid(self.low_freq_weight)
        
        fused_high = w_high * ir_high + (1 - w_high) * styled_high
        fused_low = w_low * ir_low + (1 - w_low) * styled_low
        
        return fused_high + fused_low
 
class Generator(nn.Module):
    def __init__(self, ir_in_ch=1, vis_in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
         
        self.ir_encoder = IREncoder(ir_in_ch, base_ch)
        self.decoder = UNetDecoder(out_ch, base_ch)
         
        self.style_encoder = VisibleStyleEncoder(vis_in_ch, base_ch)
         
        self.spade1 = styleSPACE(base_ch, base_ch, alpha_init=0.2)       
        self.spade2 = styleSPACE(base_ch*2, base_ch*2, alpha_init=0.5)   
        self.spade3 = styleSPACE(base_ch*4, base_ch*4, alpha_init=0.8)   
        self.spade4 = styleSPACE(base_ch*8, base_ch*8, alpha_init=0.95)  
         
        self.skip1 = skipcon(base_ch)
        self.skip2 = skipcon(base_ch*2)
        self.skip3 = skipcon(base_ch*4)
        self.skip4 = skipcon(base_ch*8)   
         
        self.skip4.high_freq_weight.data = torch.tensor(0.95)   
        self.skip4.low_freq_weight.data = torch.tensor(0.3)     
        
    def forward(self, ir_img, visible_img):
        
        [f1, f2, f3, f4] = self.ir_encoder(ir_img)
         
        [s1, s2, s3, s4] = self.style_encoder(visible_img)
         
        f1_styled = self.spade1(f1, s1)
        f2_styled = self.spade2(f2, s2)
        f3_styled = self.spade3(f3, s3)
        f4_styled = self.spade4(f4, s4)   
         
        f1_final = self.skip1(f1, f1_styled)   
        f2_final = self.skip2(f2, f2_styled)   
        f3_final = self.skip3(f3, f3_styled)   
        f4_final = self.skip4(f4, f4_styled)   
         
        out = self.decoder(f1_final, f2_final, f3_final, f4_final)
        
        debug_info = {
            'spade_alphas': [
                self.spade1.alpha.item(),
                self.spade2.alpha.item(),
                self.spade3.alpha.item(),
                self.spade4.alpha.item()
            ],
            'skip_weights': {
                'layer4_high_freq': self.skip4.high_freq_weight.item(),
                'layer4_low_freq': self.skip4.low_freq_weight.item()
            },
            'feature_shapes': {
                'f1': f1.shape, 'f2': f2.shape,
                'f3': f3.shape, 'f4': f4.shape
            }
        }
        
        return out, debug_info
 