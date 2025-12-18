# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for EHR temporal sequences from Vision Transformer

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Create 1D sinusoidal positional embeddings for temporal sequences
    
    Args:
        embed_dim: Embedding dimension
        length: Sequence length
        cls_token: Whether to include CLS token
    
    Returns:
        pos_embed: [length, embed_dim] or [1+length, embed_dim]
    """
    grid = np.arange(length, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisitEmbed(nn.Module):
    """
    Embedding layer for EHR visit sequences
    Projects raw visit features into embedding space
    """
    def __init__(self, num_features, embed_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Linear projection with additional hidden layer for better feature extraction
        self.proj = nn.Sequential(
            nn.Linear(num_features, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, F) where
               B = batch size
               L = sequence length
               F = number of features
        
        Returns:
            Tensor of shape (B, L, D) where D = embed_dim
        """
        return self.proj(x)


class TemporalTransformerPredictor(nn.Module):
    """
    Predictor for temporal EHR sequences
    Predicts masked (future) visits given context visits
    """
    def __init__(
        self,
        num_timesteps,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 1D positional embeddings for temporal sequences
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_timesteps, predictor_embed_dim),
            requires_grad=False
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            num_timesteps,
            cls_token=False
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        
        # Transformer blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_x, masks):
        """
        Args:
            x: Encoded context timesteps
            masks_x: Indices of context timesteps
            masks: Indices of target (prediction) timesteps
        """
        assert (masks is not None) and (masks_x is not None), \
            'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # Batch Size
        B = len(x) // len(masks_x)

        # Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # Concatenate mask tokens for prediction timesteps
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Forward through transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return predictions for masked timesteps
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for EHR sequences
    Adapted from Vision Transformer for temporal data
    """
    def __init__(
        self,
        num_features,
        sequence_length=20,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features_input = num_features
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        
        # Visit embedding layer (replaces PatchEmbed)
        self.visit_embed = VisitEmbed(
            num_features=num_features,
            embed_dim=embed_dim,
            dropout=drop_rate
        )
        
        # 1D positional embeddings for temporal sequences
        self.pos_embed = nn.Parameter(
            torch.zeros(1, sequence_length, embed_dim), 
            requires_grad=False
        )
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            sequence_length,
            cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks=None):
        """
        Args:
            x: Input tensor of shape (B, L, F) where
               B = batch size
               L = sequence length
               F = number of features
            masks: Optional list of mask indices
        
        Returns:
            Encoded sequence of shape (B, L, D) where D = embed_dim
        """
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Embed visits
        x = self.visit_embed(x)
        B, N, D = x.shape

        # Add positional embeddings
        # Handle variable sequence lengths
        if N <= self.sequence_length:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            # Interpolate if sequence is longer than expected
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_embed

        # Apply masks if provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


def temporal_transformer_predictor(**kwargs):
    """Create a temporal transformer predictor"""
    model = TemporalTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def temporal_transformer_tiny(num_features, sequence_length=20, **kwargs):
    """Tiny temporal transformer"""
    model = TemporalTransformer(
        num_features=num_features, sequence_length=sequence_length,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def temporal_transformer_small(num_features, sequence_length=20, **kwargs):
    """Small temporal transformer"""
    model = TemporalTransformer(
        num_features=num_features, sequence_length=sequence_length,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def temporal_transformer_base(num_features, sequence_length=20, **kwargs):
    """Base temporal transformer"""
    model = TemporalTransformer(
        num_features=num_features, sequence_length=sequence_length,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def temporal_transformer_large(num_features, sequence_length=20, **kwargs):
    """Large temporal transformer"""
    model = TemporalTransformer(
        num_features=num_features, sequence_length=sequence_length,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


TEMPORAL_TRANSFORMER_EMBED_DIMS = {
    'temporal_transformer_tiny': 192,
    'temporal_transformer_small': 384,
    'temporal_transformer_base': 768,
    'temporal_transformer_large': 1024,
}
