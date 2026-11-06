import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from torch.nn import init, Sequential
import numpy as np
import kornia
from text_semantics import CLIPSemanticGuide

class BasicConv(nn.Module): 
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2  
        layers = list()
        if transpose:
            # padding = kernel_size // 2 -1
            padding = kernel_size // 2
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class DSCResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSCResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module): 
    def __init__(self, out_channel, num_res=1):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module): 
    def __init__(self, channel, num_res=1):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DownSample(nn.Module): 
    
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class Encoder(nn.Module): 
    
    def __init__(self, num_res=1):
        super(Encoder, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),   
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
            EBlock(base_channel * 8, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),  # padding=kernel_size//2
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=1),
        ])

        self.down12 = DownSample(base_channel * 2, base_channel * 2)
        self.down23 = DownSample(base_channel * 4, base_channel * 4)
        self.down34 = DownSample(base_channel * 8, base_channel * 8)

    def forward(self, x):
        en_outputs = list()   
        en_1 = self.feat_extract[0](x)  # b,1,320,320 -> b,32,320,320   
        res1 = self.Encoder[0](en_1)  # b,32,320,320 -> b,32,320,320  
        en_outputs.append(res1)

        en_2 = self.feat_extract[1](res1)  # b,32,320,320 -> b,64,320,320   
        res2 = self.Encoder[1](en_2)  # b,64,320,320 -> b,64,320,320   
        x_12 = self.down12(res2)  # b,64,320,320 -> b,64,160,160   
        en_outputs.append(x_12)

        en_3 = self.feat_extract[2](x_12)  # b,64,160,160 -> b,128,160,160   
        res3 = self.Encoder[2](en_3)  # b,128,160,160 -> b,128,160,160   
        x_23 = self.down23(res3)  # b,128,160,160 -> b,128,80,80   
        en_outputs.append(x_23)
        
        en_4 = self.feat_extract[3](x_23)  # b,128,80,80 -> b,256,80,80   
        res4 = self.Encoder[3](en_4)  # b,256,80,80 -> b,256,80,80   
        x_34 = self.down34(res4)  # b,256,80,80 -> b,256,40,40   
        en_outputs.append(x_34)
         #  b,32,320,320 ; b,64,160,160 ; b,128,80,80 ; b,256,40,40
        
        return en_outputs

class Decoder(nn.Module): 
    
    def __init__(self, num_res=1, is_seg = False, num_classes=6):
        super(Decoder, self).__init__()
        base_channel = 32
        self.is_seg = is_seg
        self.num_classes = num_classes
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 8, num_res),  
            DBlock(base_channel * 4, num_res),  
            DBlock(base_channel * 2, num_res),  
            DBlock(base_channel, num_res)       
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)  
        ])
        self.up43 = UpSample(base_channel * 4,base_channel * 4)
        self.up32 = UpSample(base_channel * 2,base_channel * 2)
        self.up21 = UpSample(base_channel,base_channel)

        self.Skips = nn.ModuleList([
            Skip(base_channel * 15, base_channel * 1), # [32,64,128,256] 480 = 15*32
            Skip(base_channel * 15, base_channel * 2),
            Skip(base_channel * 15, base_channel * 4),
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.classifier = nn.Conv2d(base_channel, self.num_classes, kernel_size=1)
    def forward(self, inputs):
        # inputs = [b,32,320,320], [b,64,160,160], [b,128,80,80], [b,256,40,40]
        size_320 = inputs[0].shape[-2:]
        size_160 = inputs[1].shape[-2:]
        size_80 = inputs[2].shape[-2:]
        # ----------Skip Connection-----------#
        # 80x80
        i1_80 = F.interpolate(inputs[0], size=size_80, mode='bilinear', align_corners=False)
        i2_80 = F.interpolate(inputs[1], size=size_80, mode='bilinear', align_corners=False)
        i3_80 = inputs[2]
        i4_80 = F.interpolate(inputs[3], size=size_80, mode='bilinear', align_corners=False)
        skip_3 = self.Skips[2](i1_80, i2_80, i3_80, i4_80)  # -> [b,128,80,80]
        # 160x160
        i1_160 = F.interpolate(inputs[0], size=size_160, mode='bilinear', align_corners=False)
        i2_160 = inputs[1]
        i3_160 = F.interpolate(inputs[2], size=size_160, mode='bilinear', align_corners=False)
        i4_160 = F.interpolate(inputs[3], size=size_160, mode='bilinear', align_corners=False)
        skip_2 = self.Skips[1](i1_160, i2_160, i3_160, i4_160)  # -> [b,64,160,160]
        # 320x320
        i1_320 = inputs[0]
        i2_320 = F.interpolate(inputs[1], size=size_320, mode='bilinear', align_corners=False)
        i3_320 = F.interpolate(inputs[2], size=size_320, mode='bilinear', align_corners=False)
        i4_320 = F.interpolate(inputs[3], size=size_320, mode='bilinear', align_corners=False)
        skip_1 = self.Skips[0](i1_320, i2_320, i3_320, i4_320)  # -> [b,32,320,320]
        # ----------40 -> 80 -> 160 -> 320-----------#
        res0 = self.Decoder[0](inputs[3])  # b,256,40,40 -> b,256,40,40
        de_4 = self.feat_extract[0](res0)  # b,256,40,40 -> b,128,40,40
        x_43 = self.up43(de_4)             # b,128,40,40 -> b,128,80,80
        cat_3 = torch.cat([x_43, skip_3], dim=1)  # b,256,80,80
        x_3 = self.Convs[0](cat_3)                # b,128,80,80
        res1 = self.Decoder[1](x_3)               # b,128,80,80
        de_3 = self.feat_extract[1](res1)         # b,128,80,80 -> b,64,80,80
        x_32 = self.up32(de_3)                    # b,64,80,80 -> b,64,160,160
        cat_2 = torch.cat([x_32, skip_2], dim=1)  # b,128,160,160
        x_2 = self.Convs[1](cat_2)                # b,64,160,160
        res2 = self.Decoder[2](x_2)               # b,64,160,160
        de_2 = self.feat_extract[2](res2)         # b,64,160,160 -> b,32,160,160
        x_21 = self.up21(de_2)                    # b,32,160,160 -> b,32,320,320
        cat_1 = torch.cat([x_21, skip_1], dim=1)  # b,64,320,320
        x_1 = self.Convs[2](cat_1)                # b,32,320,320
        res_last = self.Decoder[3](x_1)           # b,32,320,320
        if self.is_seg:
            seg_head = self.classifier(res_last)
            return seg_head
        else:
            de_1 = self.feat_extract[-1](res_last)
            return de_1
class FFBlock(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(FFBlock, self).__init__() 
        self.conv_1_1 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*2, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channels*2, out_channels, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=True),
        )
        self.conv_1_2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=True),
        )
        self.channel_weights_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_weights_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_2 = nn.Sequential(
            BasicConv(in_channels * 2, out_channels, kernel_size=3, stride=1, relu=True),
        )


    def forward(self, en1, en2):
        cat_1_1 = torch.cat([en1, en2], dim=1)  # b,32,320,320+b,32,320,320 -> b,64,320,320
        cat_1_2 = torch.cat([en1, en2], dim=1)  # b,32,320,320+b,32,320,320 -> b,64,320,320
        shallow_conv_1_1 = self.conv_1_1(cat_1_1)  # b,64,320,320 -> b,32,320,320  
        shallow_conv_1_2 = self.conv_1_2(cat_1_2)  # b,64,320,320 -> b,32,320,320   

        channel_weights_1 = self.channel_weights_1(shallow_conv_1_1)  # b,32,320,320 -> b,32,320,320  
        channel_weights_2 = self.channel_weights_2(shallow_conv_1_2)  # b,32,320,320 -> b,32,320,320  
        x_1 = en1 * channel_weights_1
        x_2 = en2 * channel_weights_2
        cat_2 = torch.cat([x_1, x_2], dim=1)
        fus = self.conv_2(cat_2)

        return fus

class SFBlock(nn.Module): 
    def __init__(self):
        super(SFBlock, self).__init__()
        channels = [32, 64, 128, 256]
        self.body_1 = DSCResBlock(channels[0], channels[0])
        self.body_2 = DSCResBlock(channels[1], channels[1])
        self.body_3 = DSCResBlock(channels[2], channels[2])
        self.body_4 = DSCResBlock(channels[3], channels[3])
    def forward(self, inp_feats):
        seg_fea_1 = self.body_1(inp_feats[0])
        seg_fea_2 = self.body_2(inp_feats[1])
        seg_fea_3 = self.body_3(inp_feats[2])
        seg_fea_4 = self.body_4(inp_feats[3])
        return [seg_fea_1, seg_fea_2, seg_fea_3, seg_fea_4]
class SALayer(nn.Module): 
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential( 
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x*y

class FDBlock(nn.Module):
    """ Feature Decoupled block"""
    def __init__(self, in_channels=[32, 64, 128, 256]):
        super(FDBlock, self).__init__()
        self.sa_0 = SALayer(in_channels[0], reduction=4, bias=False)
        self.sa_1 = SALayer(in_channels[1], reduction=4, bias=False)
        self.sa_2 = SALayer(in_channels[2], reduction=4, bias=False)
        self.sa_3 = SALayer(in_channels[3], reduction=4, bias=False)
    def forward(self, fus_fea, modality_fea):  # B C H W
        m_0 = self.sa_0(modality_fea[0])
        x_0 = fus_fea[0] - m_0
        m_1 = self.sa_1(modality_fea[1])
        x_1 = fus_fea[1] - m_1
        m_2 = self.sa_2(modality_fea[2])
        x_2 = fus_fea[2] - m_2
        m_3 = self.sa_3(modality_fea[3])
        x_3 = fus_fea[3] - m_3
        return [x_0, x_1, x_2, x_3]
class Skip(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(Skip, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module): 
    def __init__(self, in_channels, h):
        super(SelfAttention, self).__init__()
        assert in_channels % h == 0
        self.in_channels = in_channels
        self.head_channels = in_channels // h
        self.h = h
        # key, query, value projections for all heads
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)  # query projection
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)  # key projection
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)  # value projection
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)  # output projection

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out') 
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = x.shape[:2]
        q = self.que_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b n c->b n h c/h->b h n c/h
        k = self.key_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)  # b c c/h n
        v = self.val_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b h n c/h
        att = torch.matmul(q,k) / np.sqrt(self.head_channels)  # b c n n
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)  # b c n c/h -> b n h c/h->b n c
        out = self.out_proj(out)

        return out
        
class CrossAttention(nn.Module):

    def __init__(self, in_channels, h):
        super(CrossAttention, self).__init__()
        assert in_channels % h == 0
        self.in_channels = in_channels
        self.head_channels = in_channels // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)  # query projection
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)  # key projection
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)  # value projection
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)  # output projection

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, ctf, st):
        B, N = st.shape[:2]
        st_q = self.que_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b n h c/h->b h n c/h
        ctf_k = self.key_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)   # b n h c/h->b h n c/h
        ctf_v = self.val_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)  # b n h c/h->b h c/h n
        # Self-Attention
        att = torch.matmul(ctf_k, ctf_v) / np.sqrt(self.head_channels)

        # get attention matrix
        att = torch.softmax(att, -1)
        # output
        query_res = torch.matmul(att, st_q).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)  #b h n c/h->b n h c/h-> b n c
        query_res = self.out_proj(query_res)

        return query_res

class ResidualRefineModule(nn.Module):
    def __init__(self, in_channels, high=32, width=32):
        super(ResidualRefineModule, self).__init__()
        self.in_channels = in_channels
        self.res_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
        )

    def forward(self, ctf, st):
        _ = ctf
        return st + self.res_conv(st)

class MMFNet(nn.Module):  # b,1,320,320 -
    def __init__(self, num_classes, use_clip_guidance=True, clip_model_name="ViT-B-32", clip_checkpoint_path='./check/RemoteCLIP-ViT-B-32.pt', clip_adapter_bottleneck_dim=64):
        super(MMFNet, self).__init__()
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()
        self.decoder = Decoder(is_seg=False)
        self.ffb_1 = FFBlock(in_channels=32, out_channels=32)
        self.ffb_2 = FFBlock(in_channels=64, out_channels=64)
        self.ffb_3 = FFBlock(in_channels=128, out_channels=128)
        self.ffb_4 = FFBlock(in_channels=256, out_channels=256)
        self.sfb = SFBlock()
        self.ctf_1 = CrossTaskModule(in_channels=32, high=32, width=32)  # 320
        self.ctf_2 = CrossTaskModule(in_channels=64, high=16, width=16)  # 160
        self.ctf_3 = CrossTaskModule(in_channels=128, high=8, width=8)   # 80
        self.ctf_4 = CrossTaskModule(in_channels=256, high=4, width=4)   # 40
        self.ctq_fus_1 = ResidualRefineModule(in_channels=32, high=32, width=32)
        self.ctq_seg_1 = CTQModule(in_channels=32, high=32, width=32)
        self.ctq_fus_2 = ResidualRefineModule(in_channels=64, high=16, width=16)
        self.ctq_seg_2 = CTQModule(in_channels=64, high=16, width=16)
        self.ctq_fus_3 = ResidualRefineModule(in_channels=128, high=8, width=8)
        self.ctq_seg_3 = CTQModule(in_channels=128, high=8, width=8)
        self.ctq_fus_4 = ResidualRefineModule(in_channels=256, high=4, width=4)
        self.ctq_seg_4 = CTQModule(in_channels=256, high=4, width=4)
        self.fdb_1 = FDBlock()
        self.fdb_2 = FDBlock()
        self.seghead = Decoder(is_seg=True, num_classes=num_classes)
        self.use_clip_guidance = use_clip_guidance
        if self.use_clip_guidance:
            self.clip_semantic = CLIPSemanticGuide(
                clip_model_name=clip_model_name,
                checkpoint_path=clip_checkpoint_path,
                adapter_bottleneck_dim=clip_adapter_bottleneck_dim,
                c3=128,
                c4=256,
            )
            self.clip_res_scale_3 = nn.Parameter(torch.tensor(0.1))
            self.clip_res_scale_4 = nn.Parameter(torch.tensor(0.1))
        else:
            self.clip_semantic = None
            self.register_parameter("clip_res_scale_3", None)
            self.register_parameter("clip_res_scale_4", None)

    def forward(self, ir, rgb, ir_texts=None, vis_texts=None, pv=None):
        # --------------Encoder for ir and rgb--------------#
        ir_feas = self.encoder_1(ir)  # [b,32,320,320] [b,64,160,160] [b,128,80,80] [b,256,40,40]
        rgb_feas = self.encoder_2(rgb)
        # ----------Shared Decoder for ir and rgb-----------#
        recon_ir = self.decoder(ir_feas)
        recon_rgb = self.decoder(rgb_feas)
        # -----multi-modality image feature fusion stage----#
        fus_fea_1 = self.ffb_1(ir_feas[0], rgb_feas[0])
        fus_fea_2 = self.ffb_2(ir_feas[1], rgb_feas[1])
        fus_fea_3 = self.ffb_3(ir_feas[2], rgb_feas[2])
        fus_fea_4 = self.ffb_4(ir_feas[3], rgb_feas[3])
        fus_feas = [fus_fea_1, fus_fea_2, fus_fea_3, fus_fea_4]
        # -------get seg feature----------------------------#
        seg_feas = self.sfb(fus_feas)
        if self.use_clip_guidance and (ir_texts is not None or vis_texts is not None):
            pv_for_clip = ir if pv is None else pv
            clip_sem_3, clip_sem_4 = self.clip_semantic(pv_for_clip, rgb, ir_texts, vis_texts, seg_feas[2], seg_feas[3])
            seg_feas[2] = seg_feas[2] + self.clip_res_scale_3 * clip_sem_3
            seg_feas[3] = seg_feas[3] + self.clip_res_scale_4 * clip_sem_4
        # ------------Dynamic interaction stage---------------#
        ctf_fus_1, ctf_seg_1 = self.ctf_1(fus_feas[0], seg_feas[0])
        ctf_fus_2, ctf_seg_2 = self.ctf_2(fus_feas[1], seg_feas[1])
        ctf_fus_3, ctf_seg_3 = self.ctf_3(fus_feas[2], seg_feas[2])
        ctf_fus_4, ctf_seg_4 = self.ctf_4(fus_feas[3], seg_feas[3])
        # ------------Cross-Task Query stage----------------#
        query_fus_1 = self.ctq_fus_1(ctf_fus_1, fus_feas[0])
        query_seg_1 = self.ctq_seg_1(ctf_seg_1, seg_feas[0])
        query_fus_2 = self.ctq_fus_2(ctf_fus_2, fus_feas[1])
        query_seg_2 = self.ctq_seg_2(ctf_seg_2, seg_feas[1])
        query_fus_3 = self.ctq_fus_3(ctf_fus_3, fus_feas[2])
        query_seg_3 = self.ctq_seg_3(ctf_seg_3, seg_feas[2])
        query_fus_4 = self.ctq_fus_4(ctf_fus_4, fus_feas[3])
        query_seg_4 = self.ctq_seg_4(ctf_seg_4, seg_feas[3])
        query_fus = [query_fus_1, query_fus_2, query_fus_3, query_fus_4]
        query_seg = [query_seg_1, query_seg_2, query_seg_3, query_seg_4]
        # ------------feature decoupling stage--------------#
        dec_ir = self.fdb_1(query_fus, rgb_feas)
        dec_rgb = self.fdb_2(query_fus, ir_feas)
        # -----------reconstruction decoupled ir and rgb--------------#
        recon_dec_ir = self.decoder(dec_ir)
        recon_dec_rgb = self.decoder(dec_rgb)
        # -----------seg and fus stage------------------------------#
        seg_res = self.seghead(query_seg)
        fus = self.decoder(query_fus)
        return recon_ir, recon_rgb, recon_dec_ir, recon_dec_rgb, seg_res, fus
