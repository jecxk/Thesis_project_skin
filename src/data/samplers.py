from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
