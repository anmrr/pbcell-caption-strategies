"""
Example script for synthetic blood cell image generation
using a fine-tuned Stable Diffusion model.
"""

from pathlib import Path

import torch

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
)


BASE_MODEL = "sd2-community/stable-diffusion-2-1"

FINETUNED_UNET = "path/to/unet"

OUTPUT_DIR = Path("generated_examples")


PROMPTS = {

    "neutrophil":
        "A neutrophil cell image.",

    "lymphocyte":
        "A lymphocyte cell image.",

    "monocyte":
        "A monocyte cell image.",
}


def main():

    pipe = (
        StableDiffusionPipeline
        .from_pretrained(

            BASE_MODEL,

            torch_dtype=torch.float16,

            safety_checker=None,
        )
    )

    pipe.unet = (
        UNet2DConditionModel
        .from_pretrained(

            FINETUNED_UNET,

            torch_dtype=torch.float16,
        )
    )

    pipe = pipe.to("cuda")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name, prompt in PROMPTS.items():

        class_dir = (
            OUTPUT_DIR / class_name
        )

        class_dir.mkdir(
            exist_ok=True
        )

        for idx in range(5):

            generator = (
                torch.Generator("cuda")
                .manual_seed(42 + idx)
            )

            image = pipe(

                prompt,

                generator=generator,

                num_inference_steps=30,

                height=768,

                width=768,

            ).images[0]

            image.save(
                class_dir /
                f"{class_name}_{idx}.png"
            )

    print("Generation complete.")


if __name__ == "__main__":
    main()