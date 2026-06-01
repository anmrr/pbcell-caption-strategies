"""
Dataset utilities for peripheral blood cell classification.

Supports:
- Real images from CSV splits
- Optional synthetic image augmentation
- Train / validation / test dataloaders
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from PIL import Image

from torch.utils.data import (
    DataLoader,
    Dataset,
)

import torchvision.transforms as transforms


# CELL LABELS

CELL_TYPES = [

    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

LABEL_TO_IDX = {

    cell: idx

    for idx, cell
    in enumerate(CELL_TYPES)
}

IDX_TO_LABEL = {

    idx: cell

    for cell, idx
    in LABEL_TO_IDX.items()
}


# TRANSFORMS

def get_train_transforms(
    input_size: int = 224,
) -> transforms.Compose:

    return transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(input_size),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(
    input_size: int = 224,
) -> transforms.Compose:

    return transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(input_size),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225],
        ),
    ])


# DATASET

class BloodCellDataset(Dataset):
    """
    Dataset supporting:
    - Real images from CSV metadata
    - Synthetic image directories
    """

    def __init__(

        self,

        csv_path: Optional[Path] = None,

        real_image_dir: Optional[Path] = None,

        synthetic_dirs: Optional[List[Path]] = None,

        transform: Optional[transforms.Compose] = None,
    ):

        self.transform = (
            transform
            or get_eval_transforms()
        )

        self.samples: List[
            Tuple[Path, int]
        ] = []

        # REAL IMAGES

        if csv_path and real_image_dir:

            dataframe = pd.read_csv(csv_path)

            for _, row in dataframe.iterrows():

                class_name = str(
                    row["class_name"]
                ).lower()

                if class_name not in LABEL_TO_IDX:
                    continue

                image_name = str(
                    row["image_name"]
                )

                image_path = (

                    Path(real_image_dir)

                    / class_name

                    / image_name
                )

                if image_path.exists():

                    self.samples.append(

                        (
                            image_path,
                            LABEL_TO_IDX[class_name],
                        )
                    )

        # SYNTHETIC IMAGES

        if synthetic_dirs:

            for synthetic_dir in synthetic_dirs:

                synthetic_dir = Path(
                    synthetic_dir
                )

                if not synthetic_dir.exists():
                    continue

                for class_name in CELL_TYPES:

                    class_dir = (
                        synthetic_dir /
                        class_name
                    )

                    if not class_dir.exists():
                        continue

                    for image_path in class_dir.glob("*.png"):

                        self.samples.append(

                            (
                                image_path,
                                LABEL_TO_IDX[class_name],
                            )
                        )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, int]:

        image_path, label = (
            self.samples[idx]
        )

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        if self.transform:
            image = self.transform(image)

        return image, label

    # CLASS DISTRIBUTION

    def get_class_counts(
        self,
    ) -> Dict[str, int]:

        counts = {

            cell: 0

            for cell in CELL_TYPES
        }

        for _, label in self.samples:

            counts[
                IDX_TO_LABEL[label]
            ] += 1

        return counts

    def get_class_weights(
        self,
    ) -> torch.Tensor:

        counts = self.get_class_counts()

        total = sum(counts.values())

        weights = []

        for cell in CELL_TYPES:

            if counts[cell] > 0:

                weights.append(

                    total /
                    (
                        len(CELL_TYPES)
                        * counts[cell]
                    )
                )

            else:

                weights.append(1.0)

        return torch.tensor(
            weights,
            dtype=torch.float32,
        )


# DATALOADERS

def create_dataloaders(

    train_csv: Path,

    val_csv: Path,

    test_csv: Path,

    real_image_dir: Path,

    synthetic_dirs: Optional[
        List[Path]
    ] = None,

    use_real: bool = True,

    batch_size: int = 32,

    num_workers: int = 4,

    input_size: int = 224,

) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    torch.Tensor,
]:
    """
    Build train / validation / test dataloaders.

    Validation and test splits use only real images.
    """

    train_dataset = BloodCellDataset(

        csv_path=(
            train_csv
            if use_real
            else None
        ),

        real_image_dir=(
            real_image_dir
            if use_real
            else None
        ),

        synthetic_dirs=synthetic_dirs,

        transform=get_train_transforms(
            input_size
        ),
    )

    val_dataset = BloodCellDataset(

        csv_path=val_csv,

        real_image_dir=real_image_dir,

        synthetic_dirs=None,

        transform=get_eval_transforms(
            input_size
        ),
    )

    test_dataset = BloodCellDataset(

        csv_path=test_csv,

        real_image_dir=real_image_dir,

        synthetic_dirs=None,

        transform=get_eval_transforms(
            input_size
        ),
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=True,

        drop_last=True,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,
    )

    class_weights = (
        train_dataset
        .get_class_weights()
    )

    print(
        f"Dataset sizes | "
        f"train={len(train_dataset)} | "
        f"val={len(val_dataset)} | "
        f"test={len(test_dataset)}"
    )

    return (

        train_loader,

        val_loader,

        test_loader,

        class_weights,
    )