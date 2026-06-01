# configs/sd21 - Stable Diffusion 2.1 Fine-tuning Configurations

Representative configuration files for Stable Diffusion 2.1 fine-tuning on peripheral blood cell images at 768×768 resolution.

## Quick Start

```bash
accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  -m pbcell.train.sd21_finetune_multigpu \
  --config configs/sd21/vlm_caption.yaml
```

For single-GPU:

```bash
python -m pbcell.train \
  --config configs/sd21/vlm_caption.yaml
```

---

## Available Configurations

| Config                     | Description                               |
| -------------------------- | ----------------------------------------- |
| `label_caption.yaml`  | Minimal class-label           |
| `vlm_caption.yaml`    | Vision-language generated captions        |
| `expert_caption.yaml` | Expert-refined hematological descriptions |

---

## Caption Strategies

* **Label caption**: Minimal class-only prompts.
* **VLM caption**: Automatically generated morphological descriptions using vision-language models.
* **Expert caption**: Hematology expert-refined morphological descriptions.

Certain configurations include simplified morphological terminology to reduce generation artifacts during diffusion training.

---

## Common Settings

```yaml
pretrained_model: sd2-community/stable-diffusion-2-1
resolution: 768
mixed_precision: fp16
train_batch_size: 2
gradient_accumulation_steps: 2
checkpointing_steps: 1250
seed: 42
```

Additional implementation details and quantitative evaluation are described in the associated thesis manuscript.
