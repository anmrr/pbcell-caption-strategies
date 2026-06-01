"""
Compute Fréchet Inception Distance (FID) between generated
and real peripheral blood cell images.

Supports:
- Global FID
- Per-class FID
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import models, transforms

from tqdm import tqdm


# DEVICE

def get_device(requested: str) -> str:

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"

    return "cpu"


# INCEPTION FEATURE EXTRACTOR

class InceptionV3Features(nn.Module):

    def __init__(self):

        super().__init__()

        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )

        self.blocks = nn.Sequential(

            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,

            nn.MaxPool2d(3, 2),

            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,

            nn.MaxPool2d(3, 2),

            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,

            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,

            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )

        x = self.blocks(x)

        return x.view(x.size(0), -1)


# FEATURE EXTRACTION

def extract_features(
    paths,
    model,
    device,
    batch_size=32,
):

    preprocess = transforms.Compose([

        transforms.Resize((299, 299)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    all_features = []

    for i in tqdm(
        range(0, len(paths), batch_size),
        desc="Extracting features",
    ):

        batch_paths = paths[i:i + batch_size]

        images = torch.stack([

            preprocess(
                Image.open(path).convert("RGB")
            )

            for path in batch_paths

        ]).to(device)

        with torch.no_grad():

            features = model(images)

        all_features.append(
            features.cpu().numpy()
        )

    return np.concatenate(
        all_features,
        axis=0,
    )


# FID COMPUTATION

def sqrtm(matrix, eps=1e-10):

    matrix = (matrix + matrix.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    eigenvalues = np.maximum(
        eigenvalues,
        eps,
    )

    return (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


def fid(
    mu1,
    sigma1,
    mu2,
    sigma2,
    eps=1e-6,
):

    diff = mu1 - mu2

    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps

    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    covmean = sqrtm(
        sqrtm(sigma1)
        @ sigma2
        @ sqrtm(sigma1)
    )

    return float(max(

        diff @ diff

        + np.trace(sigma1)

        + np.trace(sigma2)

        - 2 * np.trace(covmean),

        0,
    ))


def compute_fid_from_paths(
    generated_paths,
    real_paths,
    model,
    device,
):

    generated_features = extract_features(
        generated_paths,
        model,
        device,
    ).astype(np.float32)

    real_features = extract_features(
        real_paths,
        model,
        device,
    ).astype(np.float32)

    return fid(

        generated_features.mean(0),
        np.cov(generated_features, rowvar=False),

        real_features.mean(0),
        np.cov(real_features, rowvar=False),
    )


# REAL IMAGE LOADING

def load_real_by_class(
    csv_path,
    real_root,
):

    dataframe = pd.read_csv(csv_path)

    real_by_class = defaultdict(list)

    for _, row in dataframe.iterrows():

        class_name = row["class_dir"].lower()

        filename = (
            f"{class_name}_"
            f"{row['image_stem']}.png"
        )

        full_path = (
            Path(real_root)
            / class_name
            / filename
        )

        if full_path.exists():

            real_by_class[class_name].append(
                str(full_path)
            )

    return real_by_class


# GENERATED IMAGE LOADING

def load_generated_by_class(
    generated_dir,
):

    generated_by_class = defaultdict(list)

    for path in Path(generated_dir).rglob("*.png"):

        class_name = path.parent.name.lower()

        generated_by_class[class_name].append(
            str(path)
        )

    return generated_by_class


# MAIN FID PIPELINE

def compute_fid_global_and_classwise(

    generated_dir,
    real_csv,
    real_root,
    device="cpu",
):

    device = get_device(device)

    model = (
        InceptionV3Features()
        .to(device)
        .eval()
    )

    real_by_class = load_real_by_class(
        real_csv,
        real_root,
    )

    generated_by_class = load_generated_by_class(
        generated_dir
    )

    results = {
        "per_class": {}
    }

    # GLOBAL FID

    all_generated = [

        path

        for class_name in generated_by_class

        for path in generated_by_class[class_name]
    ]

    all_real = [

        path

        for class_name in real_by_class

        for path in real_by_class[class_name]
    ]

    if len(all_generated) > 1 and len(all_real) > 1:

        results["global_fid"] = compute_fid_from_paths(

            all_generated,
            all_real,
            model,
            device,
        )

    else:

        results["global_fid"] = None

    # PER-CLASS FID

    for class_name in sorted(

        set(real_by_class)
        | set(generated_by_class)
    ):

        generated = generated_by_class.get(
            class_name,
            [],
        )

        real = real_by_class.get(
            class_name,
            [],
        )

        if len(generated) < 2 or len(real) < 2:
            continue

        results["per_class"][class_name] = {

            "fid": compute_fid_from_paths(
                generated,
                real,
                model,
                device,
            ),

            "n_generated": len(generated),

            "n_real": len(real),
        }

    return results


# CLI

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generated",
        required=True,
    )

    parser.add_argument(
        "--real_root",
        required=True,
    )

    parser.add_argument(
        "--real_csv",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    results = compute_fid_global_and_classwise(

        generated_dir=args.generated,

        real_csv=args.real_csv,

        real_root=args.real_root,

        device=args.device,
    )

    print("\n=== FID RESULTS ===")

    print(
        "Global FID:",
        results["global_fid"],
    )

    for class_name, values in results["per_class"].items():

        print(

            f"{class_name}: "

            f"FID={values['fid']:.2f} "

            f"(generated={values['n_generated']}, "

            f"real={values['n_real']})"
        )

    if args.output:

        Path(args.output).write_text(
            json.dumps(results, indent=2)
        )

        print(
            "Results saved to:",
            args.output,
        )


if __name__ == "__main__":
    main()