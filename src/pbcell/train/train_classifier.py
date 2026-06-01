"""
Example classifier training script for peripheral blood cell classification.

This simplified implementation demonstrates the general workflow used
for classifier-based evaluation in the associated study.
"""

from pathlib import Path

import torch

from pbcell.classify.dataset import create_dataloaders
from pbcell.classify.models import create_classifier
from pbcell.classify.trainer import ClassifierTrainer


def main():

    # PATHS

    train_csv = Path("data/splits/train.csv")

    val_csv = Path("data/splits/val.csv")

    test_csv = Path("data/splits/test.csv")

    image_dir = Path("data/images")

    output_dir = Path("outputs/classifier")

    synthetic_dirs = None

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # MODEL

    model, input_size = create_classifier(
        num_classes=8,
        pretrained=True,
        dropout=0.2,
    )

    # DATA

    train_loader, val_loader, test_loader, class_weights = (
        create_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            real_image_dir=image_dir,
            synthetic_dirs=synthetic_dirs,
            use_real=True,
            batch_size=32,
            num_workers=4,
            input_size=input_size,
        )
    )

    # TRAINER

    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        output_dir=output_dir,
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=30,
        warmup_epochs=5,
        early_stopping_patience=7,
        device=device,
    )

    # TRAIN

    results = trainer.train()

    print("\n====================")
    print("TRAINING FINISHED")
    print("====================")

    print(
        "Best validation F1:",
        results["best_val_f1"]
    )

    print(
        "Output directory:",
        output_dir
    )

    print("Classifier checkpoint saved.")


if __name__ == "__main__":
    main()