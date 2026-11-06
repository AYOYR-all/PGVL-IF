import torch
import torch.nn as nn
 
class ResidualBlock(nn.Module): 
    def __init__(self, channels, use_norm=True):
        super().__init__()
        gn_groups = min(32, channels) if channels >= 32 else channels
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(gn_groups, channels) if use_norm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(gn_groups, channels) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += x
        return self.act(out)
    
def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1, use_norm=True, use_residual=False): 
    if use_residual and in_ch == out_ch and stride == 1:
        return ResidualBlock(out_ch, use_norm)
     
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation)]
    if use_norm:
        gn_groups = min(32, out_ch) if out_ch >= 32 else out_ch
        layers.append(nn.GroupNorm(gn_groups, out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class FeatureBranch(nn.Module):
    def __init__(self, in_ch, channels_list, strides_list, dilations_list=None, use_residual=True):
        super().__init__()
        assert len(channels_list) == len(strides_list)
        
        if dilations_list is None:
            dilations_list = [1] * len(channels_list)
        
        layers = []
        for i, (out_ch, stride, dilation) in enumerate(zip(channels_list, strides_list, dilations_list)):
            # 计算适当的padding
            padding = dilation if stride == 1 else 1
            
            layers.append(conv_block(
                in_ch, out_ch, 
                stride=stride, 
                dilation=dilation, 
                padding=padding,
                use_residual=(use_residual and i > 0)  # 第一层通常不用残差
            ))
            in_ch = out_ch
            
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features  # 返回最后一层特征和中间特征列表
 
class MultiScaleFeature(nn.Module):
    def __init__(self, ir_ch=256, vis_ch=256, fusion_ch=512):
        super().__init__()
        gn = lambda ch: min(32, ch) if ch >= 32 else ch 
        self.ir_adapt = nn.Sequential(
            nn.Conv2d(ir_ch, fusion_ch//2, 3, 2, 1),
            nn.GroupNorm(gn(fusion_ch//2), fusion_ch//2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.vis_adapt = nn.Sequential(
            nn.Conv2d(vis_ch, fusion_ch//2, 3, 1, 1),
            nn.GroupNorm(gn(fusion_ch//2), fusion_ch//2),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        kernels = [1,3,5,7]
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(fusion_ch, fusion_ch//4, k, 1, k//2) for k in kernels
        ])
        # 融合判别层
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(fusion_ch, fusion_ch//2, 3, 1, 1),
            nn.GroupNorm(gn(fusion_ch//2), fusion_ch//2),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(fusion_ch//2),
            nn.Conv2d(fusion_ch//2, 1, 3, 1, 1)
        )
        # 高分辨率patch判别
        self.high_res_patch = nn.Sequential(
            nn.Conv2d(fusion_ch, 64, 1, 1, 0),
            nn.GroupNorm(gn(64), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1, 1, 0)
        )

    def forward(self, ir_feat, vis_feat):
        ir_adapted = self.ir_adapt(ir_feat)
        vis_adapted = self.vis_adapt(vis_feat)
        fused = torch.cat([ir_adapted, vis_adapted], dim=1)
        multi_scale_feat = torch.cat([conv(fused) for conv in self.multi_scale_conv], dim=1)
        regular_logits = self.fusion_layers(multi_scale_feat)
        high_res_logits = self.high_res_patch(multi_scale_feat)
        return regular_logits, high_res_logits

 
class Discriminator(nn.Module):
    def __init__(self, ir_ch=1, vis_ch=3, base_ch=32, use_residual=True):
        super().__init__()
        # IR分支 - 关键修改：显式指定 use_residual 参数
        self.ir_branch = FeatureBranch(ir_ch, [base_ch, base_ch*2, base_ch*4, base_ch*8],
                                       [2,1,1,1], use_residual=use_residual)
        # Vis分支 - 关键修改：显式指定 use_residual 参数
        self.vis_branch = FeatureBranch(vis_ch, [base_ch, base_ch*2, base_ch*4, base_ch*8],
                                        [2,2,1,1], use_residual=use_residual)
        self.fusion = MultiScaleFeature(base_ch*8, base_ch*8, base_ch*16)

    def forward(self, ir_img, vis_img, return_features=False):
        ir_feat, ir_inter = self.ir_branch(ir_img)
        vis_feat, vis_inter = self.vis_branch(vis_img)
        regular_logits, high_res_logits = self.fusion(ir_feat, vis_feat)
        if return_features:
            return regular_logits, high_res_logits, {'ir_features': ir_inter, 'vis_features': vis_inter}
        return regular_logits, high_res_logits

 
class MultiDiscriminator(nn.Module):
    def __init__(self, ir_ch=1, vis_ch=3, base_ch=32, num_scales=3, use_residual=True):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            Discriminator(ir_ch, vis_ch, base_ch, use_residual)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, 2, 1)

    def forward(self, ir_img, vis_img, return_features=False):
        results = []
        all_features = [] if return_features else None
        ir_input, vis_input = ir_img, vis_img
        for i, disc in enumerate(self.discriminators):
            if return_features:
                reg, high, feats = disc(ir_input, vis_input, return_features=True)
                all_features.append(feats)
            else:
                reg, high = disc(ir_input, vis_input)
            results.append((reg, high))
            if i < self.num_scales - 1:
                ir_input = self.downsample(ir_input)
                vis_input = self.downsample(vis_input)
        return (results, all_features) if return_features else results