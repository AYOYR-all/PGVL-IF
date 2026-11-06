import math
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    open_clip = None
    _OPEN_CLIP_IMPORT_ERROR = exc
else:
    _OPEN_CLIP_IMPORT_ERROR = None


class Adapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)

    def forward(self, x):
        return x + self.up_proj(self.activation(self.down_proj(x)))


class ViTAdapter(nn.Module):
    def __init__(self, block: nn.Module, embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.block = block
        self.adapter = Adapter(embed_dim, bottleneck_dim)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            x = self.block(x, attn_mask=attn_mask)
        else:
            x = self.block(x)
        return self.adapter(x)


class RemoteCLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        checkpoint_path: Optional[str] ='./check/ViT-B-16.safetensors',
        adapter_bottleneck_dim: int = 64,
        use_adapters: bool = True,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip is required for CLIP semantic guidance. "
                "Please install open_clip_torch."
            ) from _OPEN_CLIP_IMPORT_ERROR

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for offline/local loading.")

        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint_path,   # 直接本地加载
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.embed_dim = self.model.visual.transformer.width

        if use_adapters:
            blocks = self.model.visual.transformer.resblocks
            num_blocks = len(blocks)
            adapter_start_layer = max(0, num_blocks - 3)
            for i in range(adapter_start_layer, num_blocks):
                blocks[i] = ViTAdapter(blocks[i], self.embed_dim, adapter_bottleneck_dim)

            for p in self.model.parameters():
                p.requires_grad = False
            for module in self.model.visual.transformer.modules():
                if isinstance(module, ViTAdapter):
                    for p in module.adapter.parameters():
                        p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model.eval()
    @staticmethod
    def _load_checkpoint(checkpoint_path: str):
        ext = os.path.splitext(checkpoint_path)[1].lower()

        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading .safetensors requires safetensors. "
                    "Please install it with: python -m pip install safetensors"
                ) from exc
            ckpt = load_file(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 兼容常见格式
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]

        if not isinstance(ckpt, dict):
            raise TypeError(
                f"Unsupported checkpoint format from {checkpoint_path}. "
                f"Expected a state_dict-like dict, got {type(ckpt)}"
            )

        new_ckpt = {}
        for k, v in ckpt.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            new_ckpt[new_key] = v

        return new_ckpt

    def _normalize_for_clip(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = torch.clamp(image, 0.0, 1.0)
        return (image - self.image_mean.to(image.device, image.dtype)) / self.image_std.to(
            image.device, image.dtype
        )

    @staticmethod
    def _grid_size_from_pos_embed(positional_embedding: torch.Tensor) -> int:
        patch_pos = positional_embedding[1:, :]
        num_patches = patch_pos.shape[0]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch token count {num_patches} is not a perfect square.")
        return grid_size

    def encode_image(self, image: torch.Tensor):
        image = self._normalize_for_clip(image)
        visual = self.model.visual

        x = visual.conv1(image)  # [B, D, H_p, W_p]
        b, d, h_p, w_p = x.shape
        x = x.reshape(b, d, -1).permute(0, 2, 1)  # [B, N, D]

        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        pos_embed = visual.positional_embedding
        cls_pos = pos_embed[:1, :]
        patch_pos = pos_embed[1:, :]

        grid_size = self._grid_size_from_pos_embed(pos_embed)
        patch_pos = patch_pos.reshape(1, grid_size, grid_size, d).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h_p, w_p), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)

        x = x + torch.cat([cls_pos.unsqueeze(0), patch_pos], dim=1)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)

        global_feat = x[:, 0]
        patch_tokens = x[:, 1:]

        if visual.proj is not None:
            global_feat = global_feat @ visual.proj
            patch_tokens = patch_tokens @ visual.proj

        return global_feat, patch_tokens, (h_p, w_p)

    def encode_text(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(text).to(device)
        return self.model.encode_text(tokens)


class CLIPSemanticGuide(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        checkpoint_path: Optional[str] = './check/ViT-B-16.safetensors',
        adapter_bottleneck_dim: int = 64,
        c3: int = 128,
        c4: int = 256,
    ):
        super().__init__()

        self.clip = RemoteCLIP(
            model_name=clip_model_name,
            checkpoint_path=checkpoint_path,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            use_adapters=True,
        )

        clip_dim = self.clip.model.text_projection.shape[1]
        self.token_fuse = nn.Linear(clip_dim * 2, clip_dim)

        self.proj_c3 = nn.Linear(clip_dim, c3)
        self.proj_c4 = nn.Linear(clip_dim, c4)

        self.cross_attn_c3 = nn.MultiheadAttention(embed_dim=c3, num_heads=4, batch_first=True)
        self.cross_attn_c4 = nn.MultiheadAttention(embed_dim=c4, num_heads=8, batch_first=True)

        self.out_c3 = nn.Conv2d(c3, c3, kernel_size=1)
        self.out_c4 = nn.Conv2d(c4, c4, kernel_size=1)

    @staticmethod
    def _text_guided_patch_weight(patch_tokens: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        patch_norm = F.normalize(patch_tokens, dim=-1)
        text_norm = F.normalize(text_feat, dim=-1).unsqueeze(1)
        logits = (patch_norm * text_norm).sum(dim=-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return patch_tokens * weights

    @staticmethod
    def _to_map(tokens: torch.Tensor, grid_hw):
        b, _, c = tokens.shape
        gh, gw = grid_hw
        return tokens.permute(0, 2, 1).contiguous().view(b, c, gh, gw)

    @staticmethod
    def _cross_refine(
        feat: torch.Tensor,
        token_map: torch.Tensor,
        cross_attn: nn.MultiheadAttention,
        out_proj: nn.Module,
    ):
        b, c, h, w = feat.shape
        gh, gw = token_map.shape[-2:]
        pooled = F.adaptive_avg_pool2d(feat, (gh, gw))
        q = pooled.flatten(2).transpose(1, 2)
        kv = token_map.flatten(2).transpose(1, 2)
        refined, _ = cross_attn(q, kv, kv, need_weights=False)
        refined = refined.transpose(1, 2).contiguous().view(b, c, gh, gw)
        refined = F.interpolate(refined, size=(h, w), mode="bilinear", align_corners=False)
        return out_proj(refined)

    def forward(
        self,
        ir: torch.Tensor,
        rgb: torch.Tensor,
        ir_texts: Union[str, List[str], None],
        vis_texts: Union[str, List[str], None],
        feat3: torch.Tensor,
        feat4: torch.Tensor,
    ):
        device = feat3.device
        self.clip = self.clip.to(device)

        if ir_texts is None and vis_texts is None:
            raise ValueError("At least one of ir_texts or vis_texts must be provided.")
        if ir_texts is None:
            ir_texts = vis_texts
        if vis_texts is None:
            vis_texts = ir_texts

        text_feat_ir = self.clip.encode_text(ir_texts).to(device)
        text_feat_vis = self.clip.encode_text(vis_texts).to(device)

        _, ir_patches, grid_hw = self.clip.encode_image(ir)
        _, rgb_patches, _ = self.clip.encode_image(rgb)

        ir_patches = ir_patches.to(device)
        rgb_patches = rgb_patches.to(device)

        ir_weighted = self._text_guided_patch_weight(ir_patches, text_feat_ir)
        rgb_weighted = self._text_guided_patch_weight(rgb_patches, text_feat_vis)
        fused_tokens = self.token_fuse(torch.cat([ir_weighted, rgb_weighted], dim=-1))

        tokens_c3 = self.proj_c3(fused_tokens)
        tokens_c4 = self.proj_c4(fused_tokens)

        map_c3 = self._to_map(tokens_c3, grid_hw)
        map_c4 = self._to_map(tokens_c4, grid_hw)

        sem_c3 = self._cross_refine(feat3, map_c3, self.cross_attn_c3, self.out_c3)
        sem_c4 = self._cross_refine(feat4, map_c4, self.cross_attn_c4, self.out_c4)

        return sem_c3, sem_c4
