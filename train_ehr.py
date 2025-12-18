# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# JEPA-EHR Training Script

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.temporal import TemporalMaskCollator, SimpleFuturePredictionMask
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.mimic_ehr import make_mimic_ehr

from src.helper_ehr import (
    load_checkpoint_ehr,
    init_model_ehr,
    init_opt_ehr
)

# --
log_timings = True
log_freq = 10
checkpoint_freq = 10  # Save every 10 epochs
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.warning('CUDA not available, using CPU')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_path = args['data']['data_path']
    sequence_length = args['data']['sequence_length']
    context_length = args['data']['context_length']
    prediction_length = args['data']['prediction_length']
    feature_columns = args['data'].get('feature_columns', None)
    drop_last = args['data']['drop_last']

    # -- MASK
    masking_strategy = args['mask']['masking_strategy']
    if masking_strategy == 'temporal':
        allow_overlap = args['mask']['allow_overlap']
        num_context_blocks = args['mask']['num_context_blocks']
        num_pred_blocks = args['mask']['num_pred_blocks']
        block_size_range = args['mask']['block_size_range']
        context_ratio = args['mask']['context_ratio']
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # Create logging folder
    os.makedirs(folder, exist_ok=True)
    
    dump = os.path.join(folder, 'params-jepa-ehr.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.1f', 'mask-context'),
                           ('%.1f', 'mask-pred'),
                           ('%d', 'time (ms)'))

    # -- init mask collator
    if masking_strategy == 'simple':
        mask_collator = SimpleFuturePredictionMask(
            sequence_length=sequence_length,
            context_length=context_length,
            prediction_length=prediction_length
        )
    elif masking_strategy == 'temporal':
        mask_collator = TemporalMaskCollator(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            context_ratio=context_ratio,
            num_context_blocks=num_context_blocks,
            num_pred_blocks=num_pred_blocks,
            allow_overlap=allow_overlap,
            block_size_range=tuple(block_size_range)
        )
    else:
        raise ValueError(f'Unknown masking strategy: {masking_strategy}')

    # -- init data-loaders/samplers
    logger.info('Loading EHR dataset...')
    dataset, data_loader, sampler = make_mimic_ehr(
        data_path=data_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        feature_columns=feature_columns
    )
    
    # Get number of features from dataset
    num_features = dataset.num_features
    logger.info(f'Dataset loaded with {num_features} features')
    
    ipe = len(data_loader)
    logger.info(f'Iterations per epoch: {ipe}')

    # -- init model
    encoder, predictor = init_model_ehr(
        device=device,
        num_features=num_features,
        sequence_length=sequence_length,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim
    )
    target_encoder = copy.deepcopy(encoder)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt_ehr(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16
    )
    
    # -- wrap with DDP if distributed
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint_ehr(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler
        )
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict() if world_size == 1 else encoder.module.state_dict(),
            'predictor': predictor.state_dict() if world_size == 1 else predictor.module.state_dict(),
            'target_encoder': target_encoder.state_dict() if world_size == 1 else target_encoder.module.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'num_features': num_features,
            'sequence_length': sequence_length
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
                logger.info(f'Saved checkpoint to {save_path.format(epoch=f"{epoch + 1}")}')

    # -- TRAINING LOOP
    logger.info('Starting training...')
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(data_loader):

            def load_data():
                # -- get sequences
                sequences = udata[0].to(device, non_blocking=True)  # (B, L, F)
                # masks_enc and masks_pred are already lists of lists of tensors
                # Convert nested structure: [[tensor], [tensor], ...] to device
                masks_1 = [[m.to(device, non_blocking=True) for m in batch_masks] for batch_masks in masks_enc]
                masks_2 = [[m.to(device, non_blocking=True) for m in batch_masks] for batch_masks in masks_pred]
                return (sequences, masks_1, masks_2)
            
            sequences, masks_enc, masks_pred = load_data()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(sequences)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(sequences, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    if world_size > 1:
                        loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                if world_size > 1:
                    grad_stats = grad_logger(encoder.module.named_parameters())
                else:
                    grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2 if torch.cuda.is_available() else 0,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
