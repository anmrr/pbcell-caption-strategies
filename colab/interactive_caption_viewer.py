"""
Interactive visualization tool for qualitative inspection
of generated peripheral blood cell captions.
"""

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

import ipywidgets as widgets

from PIL import Image

from IPython.display import (
    clear_output,
    display,
)

# CONFIGURATION

DATASET_PATH = Path(
    "example_dataset"
)

CSV_PATH = (
    "example_captions.csv"
)

CLASSES = [

    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

# LOAD DATA

dataframe = pd.read_csv(
    CSV_PATH
)

# APPLICATION STATE

current_class = CLASSES[0]

class_indices = dataframe[
    dataframe["class_name"]
    == current_class
].index.tolist()

pointer = 0

# INTERACTIVE WIDGETS

dropdown_class = widgets.Dropdown(

    options=CLASSES,

    value=current_class,

    description="Class:",
)

button_previous = widgets.Button(
    description="Previous"
)

button_next = widgets.Button(
    description="Next"
)

# HELPERS

def update_indices():

    global class_indices

    global pointer

    class_indices = dataframe[

        dataframe["class_name"]
        == current_class

    ].index.tolist()

    pointer = 0


def show_image(index):

    clear_output(wait=True)

    real_index = class_indices[index]

    row = dataframe.loc[real_index]

    image_path = (

        DATASET_PATH

        / row["class_name"]

        / row["image_name"]
    )

    plt.figure(figsize=(5, 5))

    if image_path.exists():

        image = Image.open(
            image_path
        )

        plt.imshow(image)

    else:

        plt.text(
            0.5,
            0.5,
            "Image not found",
            ha="center",
        )

    plt.title(

        f"{row['class_name']} "
        f"| {index + 1}/"
        f"{len(class_indices)}"
    )

    plt.axis("off")

    plt.show()

    print("\n--- GENERATED CAPTION ---\n")

    print(row["caption"])

    print(f"\nImage path: {image_path}")

    display(

        widgets.HBox(
            [
                button_previous,
                button_next,
            ]
        )
    )

    display(dropdown_class)

# CALLBACKS

def next_image(button):

    global pointer

    if pointer < len(class_indices) - 1:

        pointer += 1

    show_image(pointer)


def previous_image(button):

    global pointer

    if pointer > 0:

        pointer -= 1

    show_image(pointer)


def change_class(change):

    global current_class

    current_class = change["new"]

    update_indices()

    show_image(pointer)

# CONNECT EVENTS

button_next.on_click(
    next_image
)

button_previous.on_click(
    previous_image
)

dropdown_class.observe(
    change_class,
    names="value",
)

# START APPLICATION

update_indices()

show_image(pointer)
