# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Helper functions for JEPA-EHR

import logging
import sys

import torch

import src.models.temporal_transformer as tt
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint_ehr(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    """
    Load checkpoint for EHR model
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # Loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # Loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # Loading target_encoder
        if target_encoder is not None:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}')

        # Loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model_ehr(
    device,
    num_features,
    sequence_length=20,
    model_name='temporal_transformer_base',
    pred_depth=6,
    pred_emb_dim=384
):
    """
    Initialize encoder and predictor models for EHR data
    
    Args:
        device: Device to put models on
        num_features: Number of input features per visit
        sequence_length: Length of temporal sequences
        model_name: Name of model architecture
        pred_depth: Depth of predictor
        pred_emb_dim: Embedding dimension of predictor
    
    Returns:
        encoder: Temporal transformer encoder
        predictor: Temporal transformer predictor
    """
    logger.info(f'Initializing EHR model: {model_name}')
    logger.info(f'  - num_features: {num_features}')
    logger.info(f'  - sequence_length: {sequence_length}')
    
    # Initialize encoder
    encoder = tt.__dict__[model_name](
        num_features=num_features,
        sequence_length=sequence_length
    )
    
    # Initialize predictor
    predictor = tt.__dict__['temporal_transformer_predictor'](
        num_timesteps=sequence_length,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # Apply weight initialization
    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    logger.info(f'Encoder parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M')
    logger.info(f'Predictor parameters: {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M')

    return encoder, predictor


def init_opt_ehr(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-5,
    final_lr=1e-6,
    use_bfloat16=False,
    ipe_scale=1.25
):
    """
    Initialize optimizer and schedulers for EHR training
    """
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                      if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                      if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                      if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                      if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW optimizer')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
