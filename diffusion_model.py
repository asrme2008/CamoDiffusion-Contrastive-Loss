"""CamoDiffusion core model (without Gated OFM).

Adapted from the model components in projectofm.ipynb. The ATCN feeds
its four feature maps directly into the denoising network; no OFM
branches or learned gates are present.
"""








import math
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath


class DWConv(nn.Module):
    """Depthwise conv used inside the MLP. Handles the extra time-token
    (position 0) by pulling it out before reshaping to a spatial map,
    then re-attaching it after — this is what makes TTC's time token
    survive passing through every internal block."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return torch.cat([time_token, x], dim=1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Spatial-reduction attention. When sr_ratio>1, the time token is
    kept out of the spatial-reduction conv (which needs a clean HxW map)
    and re-concatenated before computing keys/values — this is the exact
    mechanism that lets the timestep influence every attention step."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """This IS the ZOE module (eq. 6). mask_chans>0 adds a second,
    zero-initialized conv branch that embeds the noisy mask and adds it
    to the image embedding — only used at Stage 1 in practice."""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                               padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                        padding=(patch_size[0] // 2, patch_size[1] // 2))
            nn.init.zeros_(self.mask_proj.weight)   
            nn.init.zeros_(self.mask_proj.bias)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        if mask is not None:
            x = x + self.mask_proj(mask)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """Standard sinusoidal timestep embedding (DDPM/ADM style)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding







import torch
import torch.nn as nn
from functools import partial

class PyramidVisionTransformerImpr(nn.Module):
    """
    This class IS the ATCN. ZOE and TTC aren't separate bolt-on modules —
    they're baked directly into this backbone's forward pass:
      - ZOE: OverlapPatchEmbed at Stage 1 gets the noisy mask (mask_chans>0)
      - TTC: a per-stage time token is prepended before the transformer
             blocks, flows through real self-attention, then popped off
    """
    def __init__(self, img_size=384, in_chans=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        
        self.time_embed = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
            for d in embed_dims
        ])

        
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                                            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0]); cur += depths[0]

        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                                            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1]); cur += depths[1]

        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                                            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2]); cur += depths[2]

        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                                            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, xt, timesteps, cond_img):
        """
        xt:        [B,1,H,W]  noisy mask  (injected via ZOE, Stage 1 only)
        timesteps: [B]        diffusion timestep
        cond_img:  [B,3,H,W]  the camouflage image (the actual backbone input)
        Returns [F1,F2,F3,F4], matching our Cell-7-style convention.
        """
        B = cond_img.shape[0]
        outs = []
        x = cond_img

        for i, (patch_embed, blocks, norm) in enumerate(zip(
            [self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4],
            [self.block1, self.block2, self.block3, self.block4],
            [self.norm1, self.norm2, self.norm3, self.norm4],
        )):
            time_token = self.time_embed[i](timestep_embedding(timesteps, self.embed_dims[i])).unsqueeze(1)

            if i == 0:
                x, H, W = patch_embed(cond_img, xt)   
            else:
                x, H, W = patch_embed(x)

            x = torch.cat([time_token, x], dim=1)      
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  
            outs.append(x)

        return outs


def build_atcn(img_size=384, pretrained=True):
    model = PyramidVisionTransformerImpr(
        img_size=img_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        drop_path_rate=0.1,
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        mask_chans=1,
    )

    if not pretrained:
        return model, "random_init (pretrained=False)"

    import timm

    pretrained_model = timm.create_model("pvt_v2_b4", pretrained=True)
    pretrained_state = pretrained_model.state_dict()
    model_state = model.state_dict()
    matched = {}

    for source_key, value in pretrained_state.items():
        target_key = source_key

        for stage in range(4):
            target_key = target_key.replace(
                f"patch_embed.{stage}.", f"patch_embed{stage + 1}."
            )
            target_key = target_key.replace(
                f"stages.{stage}.blocks.", f"block{stage + 1}."
            )
            target_key = target_key.replace(
                f"stages.{stage}.norm.", f"norm{stage + 1}."
            )

        target_key = target_key.replace(
            ".mlp.dwconv.dwconv.", ".mlp.dwconv."
        )

        if target_key in model_state and model_state[target_key].shape == value.shape:
            matched[target_key] = value

    if len(matched) < 100:
        raise RuntimeError(
            "Too few PVTv2-B4 ImageNet tensors matched the ATCN backbone "
            f"({len(matched)} tensors). Check the installed timm version and model keys."
        )

    model_state.update(matched)
    model.load_state_dict(model_state)
    return model, f"timm_pvt_v2_b4_imagenet (loaded {len(matched)} tensors)"







import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class ConvBNReLU(nn.Module):
    """GroupNorm version of mmcv's ConvModule(norm_cfg=BN) — same Conv->Norm->ReLU
    structure, but insensitive to micro-batch size (safe with gradient accumulation)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=8):
        super().__init__()
        gn_groups = groups if out_channels % groups == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AdaGNBlock(nn.Module):
    """GroupNorm + scale/shift from time embedding = AdaGN, exactly as the paper defines:
    AdaGN(h, y) = y_s * GroupNorm(h) + y_b"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """Two AdaGN blocks + residual connection; time_emb -> per-channel scale & shift."""
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim else None
        self.block1 = AdaGNBlock(dim, dim_out, groups=groups)
        self.block2 = AdaGNBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Upsample(nn.Module):
    """PixelShuffle upsampling with ICNR-style init (avoids checkerboard artifacts):
    the conv weight is initialized so all `factor^2` sub-pixel groups start identical."""
    def __init__(self, dim, dim_out=None, factor=2):
        super().__init__()
        self.factor = factor
        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * (factor ** 2), 1)
        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(factor))
        self._init_conv(conv, factor)

    def _init_conv(self, conv, factor):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // (factor ** 2), i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=factor ** 2)
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class ConvProj(nn.Module):
    """Local Emphasis-style projector: 2 conv+ReLU layers, then flatten to tokens
    (matches eq. 10's CR(CR(Fi)) before upsampling/fusion)."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
        )
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DenoisingNetwork(nn.Module):
    """
    The DN. Two parallel paths that merge:
      - "down" path: encodes xt (the noisy mask) through a conv + one AdaGN
        ResnetBlock (time-conditioned)
      - "fusion" path: takes ATCN's 4 feature maps [F1,F2,F3,F4], projects
        and progressively fuses them into one map at F1's resolution
        (eq. 10-11's Local Emphasis + Zi aggregation)
    These are concatenated, passed through an "up" path (conv + 2x PixelShuffle
    upsampling stages) back to full resolution, then a final 1x1 conv predicts
    the single-channel mask logits (x0 prediction).
    """
    def __init__(self, dims=(64, 128, 320, 512), embedding_dim=256, mask_chans=1, class_num=1):
        super().__init__()
        c1, c2, c3, c4 = dims

        self.proj_c1 = ConvProj(c1, embedding_dim)
        self.proj_c2 = ConvProj(c2, embedding_dim)
        self.proj_c3 = ConvProj(c3, embedding_dim)
        self.proj_c4 = ConvProj(c4, embedding_dim)

        self.fuse34 = ConvBNReLU(embedding_dim * 2, embedding_dim, 1)
        self.fuse2 = ConvBNReLU(embedding_dim * 2, embedding_dim, 1)
        self.fuse1 = ConvBNReLU(embedding_dim * 2, embedding_dim, 1)

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), nn.SiLU(), nn.Linear(4 * embedding_dim, embedding_dim)
        )

        
        self.down_conv_in = ConvBNReLU(mask_chans, embedding_dim, 7, stride=4, padding=3)
        self.down_resblock = ResnetBlock(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim)
        self.down_conv_out = ConvBNReLU(embedding_dim, embedding_dim, 3, padding=1)

        
        self.up_conv_in = ConvBNReLU(embedding_dim * 2, embedding_dim, 1)
        self.up1 = Upsample(embedding_dim, embedding_dim // 4, factor=2)
        self.up_conv_mid = ConvBNReLU(embedding_dim // 4, embedding_dim // 4, 3, padding=1)
        self.up2 = Upsample(embedding_dim // 4, embedding_dim // 8, factor=2)
        self.up_conv_out = ConvBNReLU(embedding_dim // 8, embedding_dim // 8, 3, padding=1)

        self.pred = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(embedding_dim // 8, class_num, kernel_size=1))

    def forward(self, atcn_feats, timesteps, xt):
        c1, c2, c3, c4 = atcn_feats
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        
        x = self.down_conv_in(xt)
        x = self.down_resblock(x, t)
        x = self.down_conv_out(x)

        
        n = c4.shape[0]
        target_hw = c1.shape[2:]

        _c4 = self.proj_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=target_hw, mode='bilinear', align_corners=False)
        _c3 = self.proj_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=target_hw, mode='bilinear', align_corners=False)
        _c2 = self.proj_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=target_hw, mode='bilinear', align_corners=False)
        _c1 = self.proj_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        z34 = self.fuse34(torch.cat([_c4, _c3], dim=1))
        z2 = self.fuse2(torch.cat([z34, _c2], dim=1))
        z1 = self.fuse1(torch.cat([z2, _c1], dim=1))

        
        x = torch.cat([z1, x], dim=1)
        x = self.up_conv_in(x)
        x = self.up1(x)
        x = self.up_conv_mid(x)
        x = self.up2(x)
        x = self.up_conv_out(x)

        return self.pred(x)   


def build_dn(mask_chans=1, class_num=1):
    return DenoisingNetwork(dims=(64, 128, 320, 512), embedding_dim=256, mask_chans=mask_chans, class_num=class_num)








import torch
import torch.nn as nn
import torch.nn.functional as F


class CamoDiffusionModel(nn.Module):
    """
    Wraps ATCN + DN into a single module with the interface the training
    loop will use: given a noisy mask xt, timestep t, and the image,
    predict the clean mask x0 (logits).
    """
    def __init__(self, img_size=384, pretrained=True):
        super().__init__()
        self.img_size = int(img_size)
        self.atcn, self.weight_source = build_atcn(img_size=img_size, pretrained=pretrained)
        self.dn = build_dn(mask_chans=1, class_num=1)

    def forward_with_features(self, xt, t, image):
        """Return predicted clean-mask logits and the four ATCN feature maps.

        The feature maps are exposed for the project's contrastive auxiliary
        loss. No Gated OFM modules or gates are used in this model.
        """
        feats = self.atcn(xt, t, image)
        x0_pred_logits = self.dn(feats, t, xt)
        return x0_pred_logits, feats

    def forward(self, xt, t, image):
        logits, _ = self.forward_with_features(xt, t, image)
        return logits


def bce_iou_loss(pred_logits, mask):
    """
    Weighted BCE + weighted IoU loss (paper's eq. 5), ported directly from
    the official repo's model/loss.py. `mask` should be in {0,1} (not {-1,1}).
    The pixel-wise weight up-weights the loss near object boundaries
    (where avg_pool2d(mask) differs most from mask itself).
    """
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')

    pred = torch.sigmoid(pred_logits)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()
