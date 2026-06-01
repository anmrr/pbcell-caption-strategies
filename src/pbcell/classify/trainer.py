"""
Classifier training utilities for peripheral blood cell evaluation.

Includes:
- Training loop
- Validation
- Early stopping
- Standard classification metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.optim import AdamW

from torch.utils.data import DataLoader

from tqdm import tqdm

from .dataset import CELL_TYPES


class ClassifierTrainer:
    """
    Train and evaluate a blood cell classifier.
    """

    def __init__(

        self,

        model: nn.Module,

        train_loader: DataLoader,

        val_loader: DataLoader,

        test_loader: DataLoader,

        class_weights: Optional[
            torch.Tensor
        ] = None,

        output_dir: Path = Path(
            "outputs/classifier"
        ),

        learning_rate: float = 3e-4,

        weight_decay: float = 1e-4,

        epochs: int = 30,

        early_stopping_patience: int = 7,

        device: str = "cuda",
    ):

        self.model = model.to(device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.test_loader = test_loader

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.device = device

        self.epochs = epochs

        self.early_stopping_patience = (
            early_stopping_patience
        )

        if class_weights is not None:
            class_weights = class_weights.to(
                device
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.best_f1 = 0.0

        self.best_epoch = 0

        self.patience_counter = 0

        self.history: Dict[
            str,
            List[float]
        ] = {

            "train_loss": [],

            "val_loss": [],

            "val_f1": [],
        }

    # TRAIN ONE EPOCH

    def train_epoch(
        self,
    ) -> Tuple[float, float]:

        self.model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
        )

        for images, labels in progress_bar:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        epoch_loss = total_loss / total

        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    # EVALUATION

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        desc: str = "Evaluating",
    ):

        self.model.eval()

        total_loss = 0.0

        all_predictions = []

        all_labels = []

        for images, labels in tqdm(
            loader,
            desc=desc,
        ):

            images = images.to(self.device)

            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

        all_predictions = np.array(
            all_predictions
        )

        all_labels = np.array(
            all_labels
        )

        metrics = {

            "loss": (
                total_loss
                / len(all_labels)
            ),

            "accuracy": float(
                accuracy_score(
                    all_labels,
                    all_predictions,
                )
            ),

            "f1_macro": float(
                f1_score(
                    all_labels,
                    all_predictions,
                    average="macro",
                )
            ),
        }

        return (
            metrics,
            all_predictions,
            all_labels,
        )

    # TRAIN LOOP

    def train(self) -> Dict:

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            print(
                f"\nEpoch "
                f"{epoch}/{self.epochs}"
            )

            train_loss, train_acc = (
                self.train_epoch()
            )

            val_metrics, _, _ = (
                self.evaluate(
                    self.val_loader,
                    desc="Validation",
                )
            )

            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "val_loss"
            ].append(
                val_metrics["loss"]
            )

            self.history[
                "val_f1"
            ].append(
                val_metrics["f1_macro"]
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Validation F1: "
                f"{val_metrics['f1_macro']:.4f}"
            )

            # EARLY STOPPING

            if (
                val_metrics["f1_macro"]
                > self.best_f1
            ):

                self.best_f1 = (
                    val_metrics["f1_macro"]
                )

                self.best_epoch = epoch

                self.patience_counter = 0

                self._save_checkpoint(
                    self.output_dir /
                    "best_model.pt"
                )

            else:

                self.patience_counter += 1

                if (
                    self.patience_counter
                    >= self.early_stopping_patience
                ):

                    print(
                        "\nEarly stopping."
                    )

                    break

        # FINAL TEST EVALUATION

        self._load_checkpoint(
            self.output_dir /
            "best_model.pt"
        )

        test_metrics, test_predictions, test_labels = (
            self.evaluate(
                self.test_loader,
                desc="Testing",
            )
        )

        classification_results = (
            classification_report(
                test_labels,
                test_predictions,
                target_names=CELL_TYPES,
                digits=4,
            )
        )

        confusion = confusion_matrix(
            test_labels,
            test_predictions,
        )

        results = {

            "best_epoch": self.best_epoch,

            "best_val_f1": self.best_f1,

            "test_metrics": test_metrics,

            "history": self.history,

            "confusion_matrix": (
                confusion.tolist()
            ),

            "classification_report": (
                classification_results
            ),
        }

        with open(
            self.output_dir /
            "results.json",
            "w",
        ) as file:

            json.dump(
                results,
                file,
                indent=2,
            )

        return results

    # CHECKPOINTS

    def _save_checkpoint(
        self,
        path: Path,
    ) -> None:

        torch.save(

            {
                "model_state_dict":
                self.model.state_dict(),
            },

            path,
        )

    def _load_checkpoint(
        self,
        path: Path,
    ) -> None:

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )