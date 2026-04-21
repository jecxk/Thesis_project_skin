"""Grad-CAM implementation for model interpretability.

Generates heatmaps showing which regions of the input image
are most important for the model's prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for any CNN model.

    Usage:
        grad_cam = GradCAM(model, target_layer)
        heatmap, pred_class, confidence = grad_cam(input_tensor)
        overlay = grad_cam.overlay(original_image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            target_class: Class index to visualize. If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            pred_class: predicted class index
            confidence: prediction confidence (softmax probability)
        """
        self.model.eval()

        # Ensure requires_grad for backward pass
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Compute weights: global average pooling of gradients
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Handle different activation dimensions (for transformers with sequence dim)
        if gradients.dim() == 2:
            # (seq_len, C) -> reshape to spatial
            seq_len, channels = gradients.shape
            h = w = int(seq_len ** 0.5)
            gradients = gradients.permute(1, 0).reshape(channels, h, w)
            activations = activations.permute(1, 0).reshape(channels, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype,
                          device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()

        return heatmap, target_class, confidence

    def release(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    @staticmethod
    def overlay(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            original_image: (H, W, 3) RGB image in [0, 255]
            heatmap: (h, w) heatmap in [0, 1]
            alpha: transparency of heatmap overlay

        Returns:
            overlay: (H, W, 3) RGB image with heatmap overlay
        """
        h, w = original_image.shape[:2]

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap (returns BGR)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = np.float32(colored_heatmap) * alpha + np.float32(original_image) * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay


def get_target_layer(model, model_name: str) -> torch.nn.Module:
    """Get the appropriate target layer for Grad-CAM based on model architecture.

    Args:
        model: The model instance
        model_name: Model name string from config

    Returns:
        The target layer (last convolutional/feature layer)
    """
    name_lower = model_name.lower()

    if 'efficientnet' in name_lower:
        # EfficientNet: last conv block before global pool
        return model.conv_head

    if 'resnet' in name_lower:
        # ResNet: last residual block
        return model.layer4[-1]

    if 'densenet' in name_lower:
        # DenseNet: last dense block + final batch norm
        return model.features.norm5

    if 'swin' in name_lower:
        # Swin Transformer: last normalization layer
        return model.layers[-1].blocks[-1].norm2

    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")
