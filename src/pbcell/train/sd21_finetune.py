"""
Example Stable Diffusion 2.1 fine-tuning script for peripheral blood cell images.

This public repository version provides a simplified implementation of the
training workflow used in the associated study.

Some internal engineering details and optimization components have been omitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

from pbcell.seed import seed_all


# CONFIGURATION

SD21_MODEL_ID = "sd2-community/stable-diffusion-2-1"


@dataclass
class TrainConfig:

    experiment_name: str = "label_caption"

    train_csv: str = "data/example_dataset.csv"

    image_dir: str = "data/images"

    caption_column: Optional[str] = "caption"

    resolution: int = 768

    pretrained_model: str = SD21_MODEL_ID

    train_batch_size: int = 2

    learning_rate: float = 1e-5

    max_train_steps: int = 1000

    seed: int = 42

    validation_prompt: str = (
        "A neutrophil cell with segmented nucleus."
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)


# DATASET

class PBCellDataset(Dataset):

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        tokenizer: CLIPTokenizer,
        caption_column: str,
        resolution: int,
    ):

        self.df = pd.read_csv(csv_path)

        self.image_dir = image_dir

        self.tokenizer = tokenizer

        self.caption_column = caption_column

        self.resolution = resolution

    def __len__(self) -> int:

        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        row = self.df.iloc[idx]

        image_path = (
            self.image_dir /
            row["class_dir"] /
            f"{row['image_id']}.png"
        )

        image = Image.open(image_path).convert("RGB")

        image = image.resize(
            (self.resolution, self.resolution)
        )

        image = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        image = image.reshape(
            self.resolution,
            self.resolution,
            3
        )

        image = image.permute(2, 0, 1) / 127.5 - 1.0

        caption = str(row[self.caption_column])

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


# COLLATE

def collate_fn(batch):

    return {
        "pixel_values": torch.stack(
            [x["pixel_values"] for x in batch]
        ),
        "input_ids": torch.stack(
            [x["input_ids"] for x in batch]
        ),
    }


# VALIDATION

def generate_validation_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    step: int,
):

    image = pipeline(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    val_dir = output_dir / "validation"

    val_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        val_dir / f"step_{step}.png"
    )


# TRAINING

def train(config: TrainConfig):

    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    seed_all(config.seed)

    output_dir = Path(
        f"outputs/{config.experiment_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model,
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model,
        subfolder="unet",
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model,
        subfolder="scheduler",
    )

    dataset = PBCellDataset(
        csv_path=Path(config.train_csv),
        image_dir=Path(config.image_dir),
        tokenizer=tokenizer,
        caption_column=config.caption_column,
        resolution=config.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
    )

    unet, optimizer, dataloader = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
    )

    unet.train()

    global_step = 0

    for batch in dataloader:

        with accelerator.accumulate(unet):

            latents = vae.encode(
                batch["pixel_values"].to(torch.float16)
            ).latent_dist.sample()

            latents = (
                latents *
                vae.config.scaling_factor
            )

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(
                latents,
                noise,
                timesteps,
            )

            encoder_hidden_states = text_encoder(
                batch["input_ids"]
            )[0]

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            loss = F.mse_loss(
                model_pred.float(),
                noise.float(),
            )

            accelerator.backward(loss)

            optimizer.step()

            optimizer.zero_grad()

        global_step += 1

        print(
            f"Step {global_step} | "
            f"Loss: {loss.item():.4f}"
        )

        if global_step % 250 == 0:

            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

            generate_validation_image(
                pipeline=pipeline,
                prompt=config.validation_prompt,
                output_dir=output_dir,
                step=global_step,
            )

        if global_step >= config.max_train_steps:
            break

    accelerator.wait_for_everyone()

    accelerator.unwrap_model(
        unet
    ).save_pretrained(
        output_dir / "final_unet"
    )

    print("Training complete.")


# CLI

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = TrainConfig.from_yaml(
        args.config
    )

    train(config)


if __name__ == "__main__":
    main()