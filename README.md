# Fine-tuning LLMs for Academic Research

Teaching materials for a hands-on University of Auckland workshop on fine-tuning language models with LoRA, run by the Centre for eResearch.

**Workshop site:** https://uoa-eresearch.github.io/fine-tuning_workshop/

## Contents

- `notebooks/`: the workshop notebooks. Run them in Google Colab on a free T4 GPU (links on the workshop site).
- `data/`: the pre-generated pirate dataset used by Notebook 1. See [data/README.md](data/README.md) for how it was made and its licence (CC BY-SA 3.0).
- `scripts/make_pirate_dataset.py`: regenerates `data/` with a teacher model on a GPU machine. Learners don't need to run it.

Adapted in part from [UoA-eResearch/ahoy_training](https://github.com/UoA-eResearch/ahoy_training).
