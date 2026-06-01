"""
Utility script to compute per-class precision,
recall, and F1-score from confusion matrices.
"""

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd


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


def compute_metrics_from_confusion(
    confusion_matrix,
):

    confusion_matrix = np.array(
        confusion_matrix
    )

    results = {}

    for idx, class_name in enumerate(
        CELL_TYPES
    ):

        tp = confusion_matrix[idx, idx]

        fp = (
            confusion_matrix[:, idx]
            .sum()
            - tp
        )

        fn = (
            confusion_matrix[idx, :]
            .sum()
            - tp
        )

        support = (
            confusion_matrix[idx, :]
            .sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (

            2 * precision * recall
            / (precision + recall)

            if (precision + recall) > 0
            else 0.0
        )

        results[class_name] = {

            "precision": float(precision),

            "recall": float(recall),

            "f1": float(f1),

            "support": int(support),
        }

    return results


def summarize_metrics(metrics):

    dataframe = pd.DataFrame(metrics).T

    return {

        "macro_precision": float(
            dataframe["precision"].mean()
        ),

        "macro_recall": float(
            dataframe["recall"].mean()
        ),

        "macro_f1": float(
            dataframe["f1"].mean()
        ),
    }


def main():

    parser = argparse.ArgumentParser(

        description=(
            "Compute classification metrics "
            "from confusion matrices."
        )
    )

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.input_json,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    output = {}

    for model_name in data["models"]:

        confusion_matrix = (

            data["models"][model_name]
            ["confusion_matrix"]
        )

        metrics = (
            compute_metrics_from_confusion(
                confusion_matrix
            )
        )

        summary = summarize_metrics(
            metrics
        )

        output[model_name] = {

            "per_class": metrics,

            "summary": summary,
        }

        print(f"\n{model_name}")

        print(
            pd.DataFrame(metrics).T
        )

        print("\nSummary:")

        print(summary)

    with open(
        args.output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved results to: "
        f"{args.output_json}"
    )


if __name__ == "__main__":
    main()