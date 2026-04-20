from __future__ import annotations

from typing import Dict, List, Tuple
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=False, grad_clip=None) -> Tuple[float, List[int], List[int]]:
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='Train', leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, all_targets, all_preds


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, amp=False) -> Tuple[float, List[int], List[int], List[str]]:
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_paths = [], [], []

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

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        all_paths.extend(paths)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, all_targets, all_preds, all_paths
