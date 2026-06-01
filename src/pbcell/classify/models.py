"""Classifier model utilities for peripheral blood cell evaluation."""

from typing import Tuple

import torch.nn as nn
from torchvision import models


def create_classifier(
    num_classes: int = 8,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, int]:
    """Create a classification model for blood cell recognition.

    Returns:
        (model, input_size)
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model, 224
from typing import Tuple

import torch.nn as nn
from torchvision import models


def create_classifier(
    num_classes: int = 8,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, int]:
    """Create an EfficientNet-B0 classifier.

    Returns:
        (model, input_size)
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model, 224
