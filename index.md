---
layout: default
title: Fine-tuning LLMs for Academic Research
---

This hands-on workshop shows researchers how to adapt a language model to their own task with parameter-efficient fine-tuning (LoRA). You prepare a dataset, fine-tune a model, and measure how it performs before and after training. You need basic Python (lists, dictionaries, functions and loops), but no machine learning experience.

By the end of the workshop you will be able to:

1. Explain how fine-tuning, prompting and retrieval-augmented generation (RAG) differ, and which suits a given research task.
2. Prepare and format a custom dataset for fine-tuning.
3. Fine-tune an LLM with a parameter-efficient method such as LoRA.
4. Choose and apply suitable methods to evaluate a fine-tuned model.

## Before the workshop

You run everything in **Google Colab** in your web browser, so there is nothing to install.

1. Make sure you can sign in to a Google account.
2. Open a notebook from the table below and check that it opens in Colab.

## How to run a notebook

1. Click **Open in Colab** next to the notebook. Sign in with your Google account if asked.
2. Choose **Runtime → Change runtime type**, select **T4 GPU**, and click **Save**.
3. Colab may warn that the notebook was not written by Google. Click **Run anyway**.
4. Run the cells from top to bottom.

Colab doesn't save your changes. To keep them, choose **File → Save a copy in Drive**.

## Workshop notebooks

| Notebook | Run in Google Colab |
|----------|---------------------|
| 01: Teach a model to talk like a pirate (Unsloth) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/01_pirate_unsloth.ipynb) |
| 02: Teach a DNA model to spot promoters (Transformers + PEFT) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/02_genomics_transformers.ipynb) |
| 03: Taking it home: export your model and use your own data | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/03_taking_it_home.ipynb) |

## If Colab runs out of GPU: use Kaggle

Free Colab has a daily GPU limit. If Colab says no GPU is available, use the Kaggle versions below. They are the same notebooks, changed to run on Kaggle.

To be ready, before the workshop:

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Verify your phone number (**Settings → Phone verification**). Kaggle doesn't allow a GPU or internet access without it.

To run a notebook:

1. Click **Open in Kaggle** next to the notebook, and sign in.
2. In the panel on the right, under **Session options**, set **Accelerator** to **GPU T4 x2** and turn **Internet** on.
3. Run the cells from top to bottom.

Kaggle has no Google Drive, so at the end of each notebook you download your work instead. Notebook 3 explains how to upload your pirate adapter again.

| Notebook | Run on Kaggle |
|----------|---------------|
| 01: Teach a model to talk like a pirate (Unsloth) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/kaggle/01_pirate_unsloth_kaggle.ipynb) |
| 02: Teach a DNA model to spot promoters (Transformers + PEFT) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/kaggle/02_genomics_transformers_kaggle.ipynb) |
| 03: Taking it home: export your model and use your own data | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/UoA-eResearch/fine-tuning_workshop/blob/main/notebooks/kaggle/03_taking_it_home_kaggle.ipynb) |

---

Developed by Dr Toby Johnson, Centre for eResearch, University of Auckland
