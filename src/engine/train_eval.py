from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.mixup import mixup_data, cutmix_data, mixup_criterion


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    amp=False,
    grad_clip=None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mixup_cutmix_prob: float = 0.0,
) -> Tuple[float, List[int], List[int]]:
    """Train for one epoch, optionally with Mixup/CutMix.

    When mixup/cutmix is active, training predictions are computed from
    the mixed labels — accuracy on training set will be approximate.
    """
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    use_mixup = mixup_alpha > 0 or cutmix_alpha > 0
    pbar = tqdm(loader, desc='Train', leave=False)

    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --- Mixup / CutMix ---
        apply_mix = use_mixup and random.random() < mixup_cutmix_prob
        if apply_mix:
            # 50/50 chance of mixup vs cutmix (if both enabled)
            if mixup_alpha > 0 and cutmix_alpha > 0:
                use_cutmix = random.random() > 0.5
            elif cutmix_alpha > 0:
                use_cutmix = True
            else:
                use_cutmix = False

            if use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if apply_mix:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if apply_mix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        # For metrics, use original targets (not mixed)
        all_targets.extend(targets.detach().cpu().tolist())
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, all_targets, all_preds


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, amp=False) -> Tuple[float, List[int], List[int], List[str], torch.Tensor]:
    """Evaluate for one epoch.

    Returns:
        epoch_loss, all_targets, all_preds, all_paths, all_probs (N x C tensor)
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_paths = [], [], []
    all_probs_list = []

    pbar = tqdm(loader, desc='Eval', leave=False)
    for images, targets, paths in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.softmax(outputs.float(), dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        all_paths.extend(paths)
        all_probs_list.append(probs.detach().cpu())
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    all_probs = torch.cat(all_probs_list, dim=0)  # (N, C)
    return epoch_loss, all_targets, all_preds, all_paths, all_probs
