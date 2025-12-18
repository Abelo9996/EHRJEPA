# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    Apply masks to input tensor
    
    Args:
        x: tensor of shape [B (batch-size), N (num-timesteps), D (feature-dim)]
        masks: list of masks, where each mask is a list containing a tensor of indices
               Format: [[tensor], [tensor], ...]
    
    Returns:
        Masked tensor with only specified timesteps kept
    """
    # Handle nested list structure from collator: [[tensor], [tensor], ...]
    processed_masks = []
    for m in masks:
        if isinstance(m, list):
            # m is a list, extract first element which should be the tensor
            if len(m) > 0 and isinstance(m[0], torch.Tensor):
                processed_masks.append(m[0])
            elif isinstance(m[0], list) and len(m[0]) > 0:
                # Double nested: [[[tensor]]]
                processed_masks.append(m[0][0])
            else:
                raise ValueError(f"Unexpected mask structure: {type(m[0])}")
        else:
            # m is already a tensor
            processed_masks.append(m)
    
    # Gather masked timesteps
    all_x = []
    for m in processed_masks:
        # m should be shape (K,) where K is number of timesteps to keep
        # We need it to be shape (1, K) to broadcast properly
        if m.dim() == 1:
            m = m.unsqueeze(0)  # (K,) -> (1, K)
        
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, K, D)
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    
    return torch.cat(all_x, dim=0)
