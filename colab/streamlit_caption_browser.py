"""
Streamlit application for interactive visualization
of generated peripheral blood cell captions.
"""

from pathlib import Path

import json

import pandas as pd

import streamlit as st

# CONFIGURATION

DATASET_PATH = Path(
    "example_dataset"
)

CSV_PATH = (
    "example_captions.csv"
)

# LOAD DATA

dataframe = pd.read_csv(
    CSV_PATH,
    low_memory=False,
)

# SAFE JSON PARSER

def safe_json(value):

    try:
        return json.loads(value)

    except Exception:

        return {}

# PARSE MORPHOLOGY

morphology = dataframe[
    "morphology_json"
].apply(safe_json)

dataframe["size"] = morphology.apply(
    lambda x: x.get("size")
)

dataframe["nucleus_shape"] = morphology.apply(
    lambda x: x.get("nucleus_shape")
)

# SIDEBAR FILTERS

selected_class = st.sidebar.selectbox(

    "Class",

    sorted(
        dataframe["class_name"]
        .unique()
    )
)

filtered = dataframe[
    dataframe["class_name"]
    == selected_class
]

selected_size = st.sidebar.multiselect(

    "Size",

    sorted(
        filtered["size"]
        .dropna()
        .unique()
    )
)

if selected_size:

    filtered = filtered[
        filtered["size"]
        .isin(selected_size)
    ]

st.write(
    f"Cells displayed: "
    f"{len(filtered)}"
)

# IMAGE GRID

columns = st.columns(4)

for index, (_, row) in enumerate(

    filtered.head(40).iterrows()
):

    with columns[index % 4]:

        image_path = (

            DATASET_PATH

            / row["class_name"]

            / row["image_name"]
        )

        if image_path.exists():

            st.image(
                str(image_path),
                width=160,
            )

        st.write(row["image_name"])

        st.write(row["caption"])

        st.caption(

            f"Size: "
            f"{row['size']}"
        )

        st.caption(

            f"Nucleus: "
            f"{row['nucleus_shape']}"
        )
