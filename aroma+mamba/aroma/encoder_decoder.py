import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
from functools import partial, wraps

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from timm.models.layers import DropPath, to_2tuple
from aroma.mamba_4D import mamba_4D

from aroma.fourier_features import MultiScaleNeRFEncoding, NeRFEncoding


# ACKNOWLEDGEMENT: code adapted from perceiver implementation by lucidrains


# 0. utils function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exists(val):
    return val is not None


def cache_fn(f):  # 函数第一次运行时会正常执行、返回结果，并把这个结果存起来；之后再调用这个函数时，就不会再重新执行函数体，而是直接返回之前保存的结果
    cache = None  # 永远只返回第一次执行的结果

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def dropout_seq(images, coordinates, mask=None, dropout=0.25):
    b, n, *_, device = *images.shape, images.device
    logits = torch.randn(b, n, device=device)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")  # 用于后续广播

    if mask is None:
        images = images[batch_indices, keep_indices]
        coordinates = coordinates[batch_indices, keep_indices]

        return images, coordinates

    else:
        images = images[batch_indices, keep_indices]
        coordinates = coordinates[batch_indices, keep_indices]
        mask = mask[batch_indices, keep_indices]

        return images, coordinates, mask


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )

    def sample(self, K=1):  # 采样
        if K == 1:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(
                device=self.mean.device
            )
            return x
        else:
            x = self.mean[None, ...].repeat([K, 1, 1, 1]) + self.std[None, ...].repeat(
                K, 1, 1, 1
            ) * torch.randn([K, *self.mean.shape]).to(device=self.mean.device)
            return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # 与标准正态分布的KL散度
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
                )
            else:  # 与另一高斯分布的KL散度
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )

    def nll(self, sample, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean

# 1. Main blocks

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, H=85, W=85):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.hydra = mamba_4D(d_model=dim, d_state=64, ssm_ratio=2.0)  # output is 4 dimensions

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = H
        self.W = W
        self.dim = dim

    def forward(self, x, decoder=False, simple_decoder=False):
        inp = x
        outp = self.drop_path(
            self.hydra(self.norm1(x).reshape(-1, self.H, self.W, self.dim)).reshape(-1, self.H * self.W, self.dim))
        if not decoder:
            x = inp + outp
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif not simple_decoder:
            x = outp
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = outp

        return x


class Patchify(nn.Module):
    def __init__(self, H_img=224, W_img=224, patch_size=3, in_chans=3, embed_dim=768):
        super(Patchify, self).__init__()
        # assert img_size % patch_size == 0
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img // patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ###########################3->4,second unfold self.patch_size[0]->self.patch_size[1]
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(4, self.patch_size[1], self.patch_size[1])
        P = x.shape[2]
        # print(x.shape,P)
        x = x.contiguous().view(B, self.in_chans, self.H * self.W, self.patch_size[0], self.patch_size[1])

        embedded_tokens = torch.einsum('ijklm,cjlm->ick', x, self.WE) + self.bE.view(1, -1, 1)
        embedded_tokens = embedded_tokens.permute(0, 2, 1).contiguous()
        embedded_tokens = self.layer_norm(embedded_tokens)

        return embedded_tokens


class DePatchify(nn.Module):
    def __init__(self, H_img=224, W_img=224, patch_size=3, embed_dim=3, in_chans=768):
        super(DePatchify, self).__init__()
        # assert img_size % patch_size == 0
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img // patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        y = x + self.bE.reshape(1, 1, -1)

        y = torch.einsum('ijk,klmn->iljmn', y, self.WE)
        y = y.reshape(y.shape[0], y.shape[1], self.H, self.W, y.shape[3], y.shape[4]).permute(0, 1, 2, 4, 3, 5)
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2] * y.shape[3],
                      y.shape[4] * y.shape[5])  # turn the latent to physics
        H, W = self.H, self.W

        return y

# 2. INR Decoder

class FourierPositionalEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim=128,
            num_freq=32,
            max_freq_log2=5,
            input_dim=2,
            base_freq=2,
            use_relu=True,
    ):
        super().__init__()

        self.nerf_embedder = NeRFEncoding(
            num_freq=num_freq,
            max_freq_log2=max_freq_log2,
            input_dim=input_dim,
            base_freq=base_freq,
            log_sampling=False,
            include_input=False,
        )

        self.linear = nn.Linear(self.nerf_embedder.out_dim, hidden_dim)
        self.use_relu = use_relu

    def forward(self, coords):
        x = self.nerf_embedder(coords)
        if self.use_relu:
            x = torch.relu(self.linear(x))
        else:
            x = self.linear(x)  # try without relu

        return x


class MultiFourierEmbedding(nn.Module):
    def __init__(
            self,
            num_freq,
            hidden_dim=128,
            log_sampling=True,
            include_input=False,
            input_dim=3,
            base_freq=2,
            scales=[3, 4, 5],
            use_pi=True,
            disjoint=True,
            use_relu=True
    ):
        super().__init__()

        self.pos_multi_scale = MultiScaleNeRFEncoding(
            num_freq,
            log_sampling=log_sampling,
            include_input=include_input,
            input_dim=input_dim,
            base_freq=base_freq,
            scales=scales,
            use_pi=use_pi,
            disjoint=disjoint,
        )

        if include_input:
            self.linear = nn.Linear(self.pos_multi_scale.out_dim_per_scale * (len(scales) + 1), hidden_dim)
        else:
            self.linear = nn.Linear(self.pos_multi_scale.out_dim_per_scale * len(scales), hidden_dim)
        self.use_relu = use_relu

    def forward(self, coords):
        x = self.pos_multi_scale(coords)
        x = x.contiguous().reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        if self.use_relu:
            x = torch.relu(self.linear(x))
        else:
            x = self.linear(x)  # try without relu

        return x


class LocalityAwareINRDecoder(nn.Module):
    def __init__(self, output_dim=1, embed_dim=16, num_scales=3, dim=128, depth=3):
        super().__init__()
        self.dim = dim
        # Define Fourier transformation, linear layers, and other components
        self.depth = depth
        layers = [nn.Linear(embed_dim, dim), nn.ReLU()]  # Input layer

        # Add intermediate layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim, output_dim))  # Output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, localized_latents):
        # we stack the different scales
        localized_latents = einops.rearrange(localized_latents, "b c h w -> b (h w) c")
        return self.mlp(localized_latents)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaLN, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_scale = nn.Linear(hidden_dim, hidden_dim)
        self.fc_shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply LayerNorm first
        x_ln = self.ln(x)
        # Compute scale and shift parameters conditioned on z
        scale = self.fc_scale(z)  # .unsqueeze(1)
        shift = self.fc_shift(z)  # .unsqueeze(1)
        # Apply AdaLN transformation
        return scale * x_ln + shift


# Residual block class
class ModulationBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ModulationBlock, self).__init__()
        self.adaln1 = AdaLN(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.adaln2 = AdaLN(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply first AdaLN and linear transformation
        residual = x
        out = self.adaln1(x, z)
        out = self.silu(self.linear1(out))
        # Apply second AdaLN and linear transformation
        out = self.adaln2(out, z)
        out = self.linear2(out)
        # Residual connection
        return out + residual


# not used  but looks interesting
class LocalityAwareINRDecoderWithModulation(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=2):
        super(LocalityAwareINRDecoderWithModulation, self).__init__()
        # Stack residual blocks
        self.blocks = nn.ModuleList(
            [ModulationBlock(hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x, z):
        # Pass through each residual block
        for block in self.blocks:
            x = block(x, z)
        return x


# 3. Perceiver Encoder
class PerceiverEncoder(nn.Module):
    def __init__(
            self,
            *,
            space_dim=2,  # coordiate dimension
            val_dim=1,  # value channel
            out_dim=1,
            hidden_dim=256,  # dimension after patchify
            mlp_ratios=4,  # control hidden dim in mlp
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=torch.nn.LayerNorm,
            patch_sizes=2,
            H=64,  # input dimension after padding
            W=64,
            depth=3,
            latent_dim=16,
            max_pos_encoding_freq=4,
            num_freq=12,
            scales=[3, 4, 5],
            bottleneck_index=0,
            encode_geo=True,
            include_multi_pos=True,
            depatch_each_step=True,
            simple_decoder=True,
            use_mamba_decoder=True
    ):
        super().__init__()
        self.depth = depth
        self.bottleneck_index = bottleneck_index  # where to put the botleneck, by default 0 means just after cross attention
        self.encode_geo = encode_geo
        self.include_multi_pos = include_multi_pos
        self.depatch_each_step = depatch_each_step
        self.H = H
        self.W = W
        self.hidden_dim = hidden_dim
        self.simple_decoder = simple_decoder
        self.use_mamba_decoder = use_mamba_decoder

        # patch
        if encode_geo:
            self.patch_geo_embed = Patchify(H_img=H,
                                            W_img=W,
                                            patch_size=patch_sizes,
                                            in_chans=hidden_dim,
                                            embed_dim=hidden_dim)
            num_patches = (H // patch_sizes) * (W // patch_sizes)
            self.pos_embed_pos = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            self.patch_drop_pos = nn.Dropout(p=drop_rate)
            if depatch_each_step:
                self.depatch_geo_embed = DePatchify(H_img=H,
                                                    W_img=W,
                                                    patch_size=patch_sizes,
                                                    embed_dim=hidden_dim,
                                                    in_chans=hidden_dim)
                self.patch_val_embed = Patchify(H_img=H,
                                                W_img=W,
                                                patch_size=patch_sizes,
                                                in_chans=hidden_dim * 2,
                                                embed_dim=hidden_dim)
            else:
                self.patch_val_embed = Patchify(H_img=H,
                                                W_img=W,
                                                patch_size=patch_sizes,
                                                in_chans=hidden_dim,
                                                embed_dim=hidden_dim)
                self.merge_geo = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.patch_val_embed = Patchify(H_img=H,
                                            W_img=W,
                                            patch_size=patch_sizes,
                                            in_chans=hidden_dim * 2,
                                            embed_dim=hidden_dim)
        num_patches_val = (H // patch_sizes) * (W // patch_sizes)
        self.pos_embed_val = nn.Parameter(torch.zeros(1, num_patches_val, hidden_dim))
        self.patch_drop_val = nn.Dropout(p=drop_rate)

        if include_multi_pos:
            if depatch_each_step:
                self.depatch_layers_embed = DePatchify(H_img=H,
                                                       W_img=W,
                                                       patch_size=patch_sizes,
                                                       embed_dim=hidden_dim,
                                                       in_chans=hidden_dim)
                self.patch_multi_embed = Patchify(H_img=H,
                                                  W_img=W,
                                                  patch_size=patch_sizes,
                                                  in_chans=hidden_dim * 2,
                                                  embed_dim=hidden_dim)
            else:
                self.patch_multi_embed = Patchify(H_img=H,
                                                  W_img=W,
                                                  patch_size=patch_sizes,
                                                  in_chans=hidden_dim,
                                                  embed_dim=hidden_dim)
                self.merge_multi = nn.Linear(hidden_dim * 2, hidden_dim)
            num_patches_multi = (H // patch_sizes) * (W // patch_sizes)
            self.pos_embed_multi = nn.Parameter(torch.zeros(1, num_patches_multi, hidden_dim))
            self.patch_drop_multi = nn.Dropout(p=drop_rate)

        self.depatch_final = DePatchify(H_img=H,
                                        W_img=W,
                                        patch_size=patch_sizes,
                                        embed_dim=hidden_dim,
                                        in_chans=hidden_dim)

        # embed coords
        self.pos_encoding = FourierPositionalEmbedding(hidden_dim=hidden_dim,
                                                       num_freq=num_freq,
                                                       max_freq_log2=max_pos_encoding_freq,
                                                       input_dim=space_dim,
                                                       base_freq=2,
                                                       use_relu=True)

        self.pos_multi_scale = MultiFourierEmbedding(num_freq=num_freq,
                                                     log_sampling=True,
                                                     include_input=False,
                                                     input_dim=space_dim,
                                                     base_freq=2,
                                                     scales=scales,
                                                     use_pi=True,
                                                     disjoint=True,
                                                     use_relu=True)

        self.lift_values = nn.Linear(val_dim, hidden_dim)

        # just extract coords features
        self.encode_geo_mamba = MambaBlock(dim=hidden_dim, mlp_ratio=mlp_ratios, drop=drop_rate,
                                           drop_path=drop_path_rate,
                                           norm_layer=norm_layer, H=H // patch_sizes, W=W // patch_sizes)

        self.encode_val_mamba = MambaBlock(dim=hidden_dim, mlp_ratio=mlp_ratios, drop=drop_rate,
                                           drop_path=drop_path_rate,
                                           norm_layer=norm_layer, H=H // patch_sizes, W=W // patch_sizes)

        # coords and value
        get_mamba_layer = lambda: MambaBlock(dim=hidden_dim, mlp_ratio=mlp_ratios, drop=drop_rate,
                                             drop_path=drop_path_rate,
                                             norm_layer=norm_layer, H=H // patch_sizes, W=W // patch_sizes)
        get_mamba_layer = cache_fn(get_mamba_layer)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(get_mamba_layer())

        # decoder
        self.decoder_mamba = MambaBlock(dim=hidden_dim, mlp_ratio=mlp_ratios, drop=drop_rate,
                                        drop_path=drop_path_rate,
                                        norm_layer=norm_layer, H=H // patch_sizes, W=W // patch_sizes)

        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim)
        self.lift_z = nn.Linear(latent_dim, hidden_dim)

    def forward(
            self,
            images,
            coords,
            mask=None,
            target_coords=None,
            sample_posterior=True,
            return_stats=False,
    ):
        b, *_, device = *images.shape, images.device

        pos_embed = self.pos_encoding(coords).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1, 2)
        val_embed = self.lift_values(images).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1, 2)
        pos_multi = self.pos_multi_scale(coords).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1,
                                                                                                              2)

        if self.encode_geo:
            pos_patch = self.patch_geo_embed(pos_embed)
            pos = self.patch_drop_pos(pos_patch + self.pos_embed_pos)
            pos = self.encode_geo_mamba(pos)
            if self.depatch_each_step:
                pos = self.depatch_geo_embed(pos)
                x = torch.cat((pos, val_embed), -1)
                x = self.patch_val_embed(x)
                x = self.patch_drop_val(x + self.pos_embed_val)
            else:
                val = self.patch_val_embed(val_embed)
                val = self.patch_drop_val(val + self.pos_embed_val)
                x = torch.cat((pos, val), -1)
                x = self.merge_geo(x)
        else:
            x = torch.cat((pos_embed, val_embed), -1)
            x = self.patch_val_embed(x)
            x = self.patch_drop_val(x + self.pos_embed_val)
        x = self.encode_val_mamba(x)

        # layers
        for index, self_mamba in enumerate(self.layers):
            if index == self.bottleneck_index:
                # bottleneck
                mu = self.mean_fc(x)
                logvar = self.logvar_fc(x)
                posterior = DiagonalGaussianDistribution(mu, logvar)

                if sample_posterior:
                    z = posterior.sample()
                else:
                    z = posterior.mode()

                x = self.lift_z(z)

            x = self_mamba(x)

        if self.bottleneck_index == len(self.layers):
            # bottleneck
            mu = self.mean_fc(x)
            logvar = self.logvar_fc(x)
            posterior = DiagonalGaussianDistribution(mu, logvar)

            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

            x = self.lift_z(z)

        # mamba from decoder multi position to latents
        if self.use_mamba_decoder:
            if self.include_multi_pos:
                if self.depatch_each_step:
                    x = self.depatch_layers_embed(x)
                    x = torch.cat((pos_multi, x), -1)
                    x = self.patch_multi_embed(x)
                    x = self.patch_drop_multi(x + self.pos_embed_multi)
                else:
                    multi = self.patch_multi_embed(pos_multi)
                    multi = self.patch_drop_multi(multi + self.pos_embed_multi)
                    x = torch.cat((multi, x), -1)
                    x = self.merge_multi(x)
            x = self.decoder_mamba(x, decoder=True, simple_decoder=self.simple_decoder)
        latents = self.depatch_final(x)

        # final linear out
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if return_stats:
            return latents, kl_loss, mu, logvar

        return latents, kl_loss

    def get_features(self, images, coords, mask=None):
        b, *_, device = *images.shape, images.device

        c = coords.clone()

        pos_embed = self.pos_encoding(c).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1, 2)
        val_embed = self.lift_values(images).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1, 2)

        if self.encode_geo:
            pos_patch = self.patch_geo_embed(pos_embed)
            pos = self.patch_drop_pos(pos_patch + self.pos_embed_pos)
            pos = self.encode_geo_mamba(pos)
            if self.depatch_each_step:
                pos = self.depatch_geo_embed(pos)
                x = torch.cat((pos, val_embed), -1)
                x = self.patch_val_embed(x)
                x = self.patch_drop_val(x + self.pos_embed_val)
            else:
                val = self.patch_val_embed(val_embed)
                val = self.patch_drop_val(val + self.pos_embed_val)
                x = torch.cat((pos, val), -1)
                x = self.merge_geo(x)
        else:
            x = torch.cat((pos_embed, val_embed), -1)
            x = self.patch_val_embed(x)
            x = self.patch_drop_val(x + self.pos_embed_val)
        x = self.encode_val_mamba(x)

        # layers
        for index, self_mamba in enumerate(self.layers):
            if index == self.bottleneck_index:
                # bottleneck
                mu = self.mean_fc(x)
                logvar = self.logvar_fc(x)
                return mu, logvar
            x = self_mamba(x)

        if self.bottleneck_index == len(self.layers):
            # bottleneck
            mu = self.mean_fc(x)
            logvar = self.logvar_fc(x)
            return mu, logvar

    def process(self, features, coords):
        pos_multi = self.pos_multi_scale(coords).reshape(-1, self.H, self.W, self.hidden_dim).permute(0, 3, 1, 2)
        x = features

        # cross attend from decoder queries to latents
        for index, self_mamba in enumerate(self.layers):
            if self.bottleneck_index == index:
                x = self.lift_z(features)
                x = self_mamba(x)
            elif self.bottleneck_index > index:
                pass

            else:
                x = self_mamba(x)

        if self.bottleneck_index == len(self.layers):
            x = self.lift_z(features)

        if self.use_mamba_decoder:
            if self.include_multi_pos:
                if self.depatch_each_step:
                    x = self.depatch_layers_embed(x)
                    x = torch.cat((pos_multi, x), -1)
                    x = self.patch_multi_embed(x)
                    x = self.patch_drop_multi(x + self.pos_embed_multi)
                else:
                    multi = self.patch_multi_embed(pos_multi)
                    multi = self.patch_drop_multi(multi + self.pos_embed_multi)
                    x = torch.cat((multi, x), -1)
                    x = self.merge_multi(x)
            x = self.decoder_mamba(x, decoder=True, simple_decoder=self.simple_decoder)
        latents = self.depatch_final(x)

        # final linear out
        return latents

class AROMAEncoderDecoderKL(nn.Module):
    def __init__(
            self,
            space_dim=2,  # coordiate dimension shape (B, L, C)
            val_dim=1,  # value channel (B, L, C)
            out_dim=1,
            hidden_dim=256,  # dimension after patchify
            mlp_ratios=4,  # control hidden dim in mlp
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=torch.nn.LayerNorm,
            patch_sizes=2,
            H=64,  # input dimension after padding
            W=64,
            num_mamba=3,
            latent_dim=8,
            max_pos_encoding_freq=4,
            num_freq=12,
            scales=[3, 4, 5],
            bottleneck_index=0,
            encode_geo=True,
            include_multi_pos=True,
            depatch_each_step=True,
            simple_decoder=True,
            use_mamba_decoder=True,

            decoder_dim=64,
            depth_inr=3,
    ):
        super().__init__()

        self.encoder = PerceiverEncoder(
            space_dim=space_dim,  # coordiate dimension
            val_dim=val_dim,  # value channel
            out_dim=out_dim,
            hidden_dim=hidden_dim,  # dimension of every token after patchify
            mlp_ratios=mlp_ratios,  # control hidden dim in mlp
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_sizes=patch_sizes,
            H=H,  # input dimension after padding
            W=W,
            depth=num_mamba,  # depth of net
            latent_dim=latent_dim,  # latent dimension of the reduced representation in VAE
            max_pos_encoding_freq=max_pos_encoding_freq,  # maximum frequency embedding for encoding the pixels
            num_freq=num_freq,  # number of frequencies for the positional embedding
            scales=scales,
            bottleneck_index=bottleneck_index,  # index of the bottleneck layer
            encode_geo=encode_geo,
            include_multi_pos=include_multi_pos,
            depatch_each_step=depatch_each_step,
            simple_decoder=simple_decoder,
            use_mamba_decoder=use_mamba_decoder
        )

        self.decoder = LocalityAwareINRDecoder(
            output_dim=out_dim,
            embed_dim=hidden_dim,
            num_scales=len(scales),
            dim=decoder_dim,
            depth=depth_inr,
        )

    def forward(
            self,
            images,
            coords,
            mask=None,
            target_coords=None,
            return_stats=False,
            sample_posterior=True,
    ):
        if return_stats:
            localized_latents, kl_loss, mean, logvar = self.encoder(
                images,
                coords,
                mask,
                target_coords,
                return_stats=return_stats,
                sample_posterior=sample_posterior,
            )
        else:
            localized_latents, kl_loss = self.encoder(
                images,
                coords,
                mask,
                target_coords,
                return_stats=return_stats,
                sample_posterior=sample_posterior,
            )

        output_features = self.decoder(localized_latents)

        if return_stats:
            return output_features, kl_loss, mean, logvar

        return output_features, kl_loss

    def encode(self, images, coords, mask=None):
        mu, logvar = self.encoder.get_features(images, coords, mask)

        return mu, logvar

    def decode(self, features, coords):
        localized_latents = self.encoder.process(features, coords)
        output_features = self.decoder(localized_latents)

        return output_features
