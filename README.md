# Stable Diffusion Model for Peripheral Blood Cell Image Generation

Fine-tuning of Stable Diffusion 2.1 for synthetic peripheral blood cell image generation using different semantic conditioning strategies.

This repository contains a minimal public version of the framework used to explore how caption semantic richness influences biomedical diffusion model performance in hematological microscopy.

---

# Overview

The project evaluates three caption conditioning strategies for diffusion training:

| Strategy              | Description                                    |
| --------------------- | ---------------------------------------------- |
| Label captions        | Minimal class-only prompts                     |
| VLM captions          | Automatically generated morphological captions |
| Hematologist captions | Expert-designed morphology captions            |

The goal is to study how semantic conditioning affects:

* image realism,
* morphological consistency,
* class fidelity,
* and downstream classifier utility.

---

# Cell Types

The framework supports 8 peripheral blood cell classes:

* Basophil
* Eosinophil
* Erythroblast
* Immature granulocyte (IG)
* Lymphocyte
* Monocyte
* Neutrophil
* Platelet

---

# Repository Structure

```text
pbcell-caption-strategies/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── accelerate/
│   │   └── dual_gpu.yaml
│   │
│   └── sd21/
│       ├── README.md
│       ├── expert_caption.yaml
│       ├── vlm_caption.yaml
│       └── label_caption.yaml
│
├── src/
│   └── pbcell/
│       ├── data.py
│       ├── seed.py
│       │
│       ├── captions/
│       │   ├── captions.py
│       │   ├── kb_load.py
│       │   └── normalization.py
│       │
│       ├── train/
│       │   ├── sd21_finetune.py
│       │   ├── train_classifier.py
│       │   └── train_modal_example.py
│       │
│       ├── evaluate/
│       │   ├── fid.py
│       │   └── oracle.py
│       │
│       └── classify/
│           ├── dataset.py
│           ├── models.py
│           └── trainer.py
│
├── scripts/
│   ├── compute_precision_recall.py
│   └── generate_images.py
│
├── colab/
│   ├── interactive_caption_viewer.py
│   └── streamlit_caption_browser.py
│
├── r_analysis/
│   └── stats.Rmd
│
├── data/
│   ├── processed/
│   │   ├── example_captions.csv
│   │   └── README.md
│   │
│   └── splits/
│       └── split_dataset.py
│
└── spec/
    ├── README.md
    └── morphology_kb.json
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/anmrr/pbcell-caption-strategies.git

cd pbcell-caption-strategies
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Example single-GPU training:

```bash
export PYTHONPATH=src

python -m pbcell.train.main \
  --config configs/sd21/label_caption.yaml
```

Example multi-GPU training with Accelerate:

```bash
export PYTHONPATH=src

accelerate launch \
  --config_file configs/accelerate/dual_gpu.yaml \
  -m pbcell.train.main \
  --config configs/sd21/vlm_caption.yaml
```

---

# Evaluation

Two evaluation approaches are included:

| Metric            | Purpose                          |
| ----------------- | -------------------------------- |
| FID               | Distribution-level image quality |
| Oracle classifier | Morphological/class fidelity     |

Example:

```bash
python -m pbcell.evaluate.main fid
```

```bash
python -m pbcell.evaluate.main oracle
```

---

# Results Overview

| Strategy              | Main Observation                       |
| --------------------- | -------------------------------------- |
| Label captions        | Highest Oracle classification fidelity |
| VLM captions          | Improved semantic richness             |
| Hematologist captions | Better morphology interpretability     |

---

# Example Dataset

The `data/` directory contains a small example dataset for demonstration purposes.

The complete peripheral blood cell dataset and trained diffusion checkpoints are not included in the public repository version.

The example CSV demonstrates the expected structure for:

* image organization,
* caption strategies,
* and morphology annotations.

---

## Dataset Source

This project uses peripheral blood smear images derived from the public dataset introduced by:

Acevedo, A. et al. *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*. Data in Brief, 30, 105474 (2020). https://doi.org/10.1016/j.dib.2020.105474

The original dataset contains annotated MGG-stained peripheral blood cell images acquired using a Cellavision DM96 microscope system.

---

# Interactive Visualization Tools

The `colab/` directory contains lightweight interactive utilities for:

* qualitative caption inspection,
* morphology browsing,
* image filtering,
* and semantic validation workflows.

These scripts are simplified public examples derived from the internal annotation and review environment used during dataset curation.

---

# Statistical Analysis

The `r_analysis/` directory contains simplified R examples for:

* FID visualization,
* Oracle metric visualization,
* heatmaps,
* and statistical comparisons.

The public version includes lightweight reproducible examples rather than the complete experimental analysis pipeline.

---

# Notes

* This repository contains a minimal reproducible research version.
* Internal experimental workflows, full datasets, and trained checkpoints are intentionally omitted.
* Example scripts are provided for educational and methodological purposes.

---

# Citation

```bibtex
@misc{pbcell_diffusion_2026,
  title={Stable Diffusion Model for Peripheral Blood Cell Image Generation},
  author={Rosero, Angie},
  year={2026}
}
```

---

# License

Apache-2.0
