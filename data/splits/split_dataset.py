"""
Utility script to create validation and test splits
from real peripheral blood cell images while avoiding
overlap with training samples.
"""

import argparse
import random

from pathlib import Path

import pandas as pd


def split_real_dataset(

    train_csv: Path,

    real_root: Path,

    output_dir: Path,

    val_ratio: float = 0.5,

    seed: int = 42,
):
    """
    Create validation and test splits using images
    not present in the training split.
    """

    random.seed(seed)

    train_dataframe = pd.read_csv(
        train_csv
    )

    used_ids = set(

        train_dataframe[
            "image_id"
        ].astype(str)
    )

    real_root = Path(real_root)

    image_pool = []

    # BUILD IMAGE POOL

    for image_path in real_root.rglob("*.png"):

        image_id = image_path.stem

        class_name = (
            image_path.parent.name
            .lower()
        )

        if image_id not in used_ids:

            image_pool.append(

                (
                    str(image_path),
                    class_name,
                )
            )

    random.shuffle(image_pool)

    # SPLIT

    split_index = int(
        len(image_pool)
        * val_ratio
    )

    validation_split = (
        image_pool[:split_index]
    )

    test_split = (
        image_pool[split_index:]
    )

    # SAVE

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(

        validation_split,

        columns=[
            "path",
            "class_name",
        ],

    ).to_csv(

        output_dir / "val.csv",

        index=False,
    )

    pd.DataFrame(

        test_split,

        columns=[
            "path",
            "class_name",
        ],

    ).to_csv(

        output_dir / "test.csv",

        index=False,
    )

    print(
        f"Pool size: {len(image_pool)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_split)}"
    )

    print(
        f"Test images: "
        f"{len(test_split)}"
    )


def main():

    parser = argparse.ArgumentParser(

        description=(
            "Create validation and test "
            "splits without train overlap."
        )
    )

    parser.add_argument(
        "--train_csv",
        required=True,
    )

    parser.add_argument(
        "--real_root",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    split_real_dataset(

        train_csv=Path(args.train_csv),

        real_root=Path(args.real_root),

        output_dir=Path(args.output_dir),

        val_ratio=args.val_ratio,

        seed=args.seed,
    )


if __name__ == "__main__":
    main()