# Public repository example implementation.
# Internal experimental details omitted.

"""
Example VLM-based caption generation pipeline for peripheral blood cell images.

This simplified implementation demonstrates the general workflow used for
morphological caption generation in the associated study.

Some internal prompting logic and optimization details have been omitted
from the public repository version.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


# CONTROLLED MORPHOLOGY SCHEMA

REQUIRED_FIELDS = [
    "cell_size",
    "nucleus_shape",
    "chromatin_pattern",
    "cytoplasm_characteristics",
    "granulation",
]


# SYSTEM PROMPT

SYSTEM_INSTRUCTION = """
You are an expert hematological morphology assistant.

Generate concise and structured peripheral blood cell
descriptions using controlled morphology terminology.
"""


# IMAGE LOADING

def load_image(path: Path) -> Image.Image:
    """Load image."""

    return Image.open(path).convert("RGB")



# PROMPT CONSTRUCTION

def build_prompt(
    cell_type: str | None = None
) -> str:
    """
    Build simplified morphology prompt.
    """

    prompt = (
        "Analyze the peripheral blood cell image and "
        "generate a concise morphology description."
    )

    if cell_type:
        prompt += f"\nCell class: {cell_type}"

    return prompt


# RESPONSE SCHEMA

def build_response_schema() -> Dict[str, List[str]]:
    """
    Example structured response schema.
    """

    return {
        "required_fields": REQUIRED_FIELDS
    }


# VLM PLACEHOLDER

async def generate_caption(
    image: Image.Image,
    prompt: str
) -> Dict[str, str]:
    """
    Example asynchronous VLM inference placeholder.
    """

    await asyncio.sleep(0.1)

    return {
        "cell_size": "intermediate",
        "nucleus_shape": "segmented",
        "chromatin_pattern": "condensed",
        "cytoplasm_characteristics": "moderate cytoplasm",
        "granulation": "fine azurophilic granules"
    }


# MORPHOLOGY NORMALIZATION

def normalize_morphology(
    morphology: Dict[str, str]
) -> Dict[str, str]:
    """
    Normalize morphology terminology.
    """

    return {
        k: v.strip().lower()
        for k, v in morphology.items()
    }


# CAPTION RENDERING

def render_caption(
    morphology: Dict[str, str]
) -> str:
    """
    Convert morphology dictionary into caption text.
    """

    return (
        f"A cell with {morphology['cell_size']} size, "
        f"{morphology['nucleus_shape']} nucleus, "
        f"{morphology['chromatin_pattern']} chromatin, "
        f"{morphology['cytoplasm_characteristics']}, "
        f"and {morphology['granulation']}."
    )


# MAIN

async def async_main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--cell-type",
        type=str,
        default=None
    )

    args = parser.parse_args()

    image = load_image(args.image)

    prompt = build_prompt(args.cell_type)

    schema = build_response_schema()

    morphology = await generate_caption(
        image=image,
        prompt=prompt
    )

    morphology = normalize_morphology(morphology)

    caption = render_caption(morphology)

    print(json.dumps(schema, indent=2))
    print(caption)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()