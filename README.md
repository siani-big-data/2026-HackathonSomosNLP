<div align="center">

# ⬜🟦🟨 1xe ⬜🟦🟨

### Canary-style Spanish assistant with post-training and retrieval grounding

**Made in the Canary Islands**

[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers 4.57+](https://img.shields.io/badge/Transformers-4.57%2B-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers/)
[![Qwen 2.5](https://img.shields.io/badge/Qwen-2.5--7B-5B6CFF?style=for-the-badge)](https://huggingface.co/Qwen)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-10B981?style=for-the-badge)](https://github.com/huggingface/peft)
[![RAG](https://img.shields.io/badge/RAG-local%20sqlite%20fts5-7C3AED?style=for-the-badge)](#training-2-style--rag)
[![uv managed](https://img.shields.io/badge/uv-managed-222222?style=for-the-badge)](https://docs.astral.sh/uv/)
[![Status prototype](https://img.shields.io/badge/status-research%20prototype-8B5CF6?style=for-the-badge)](#current-limitations)

1xe is a research prototype for building a Canary Islands Spanish assistant with a recognizable local voice, post-training over conversation datasets, and optional RAG over curated cultural sources such as Academia Canaria, Canariwiki, and GEVIC.

</div>

## Overview

This repository combines three main pieces:

- data collection and cleaning for Canary cultural sources
- Qwen post-training to learn conversational Canary style
- a second post-training stage with RAG context for better grounded answers

The currently active parts of the project are:

- [`siani/data_preparation/`](siani/data_preparation/)
- [`siani/post_training/`](siani/post_training/)

## Expected structure

```text
siani/
  data/
    academia_canaria/
    canariwiki/
    gevic/
    post/
      canary_style.jsonl
      *.jsonl
  data_preparation/
  post_training/
outputs/
```

Note: the current repository layout still uses the `siani/` package path internally. `1xe` is the project name, while `siani/...` is the existing code path in this repo.

### Key directories

- `siani/data/post/`
  Conversational post-training datasets in `jsonl` format.

- `siani/data/academia_canaria/`
- `siani/data/canariwiki/`
- `siani/data/gevic/`
  Knowledge sources used by the RAG layer.

- `outputs/`
  LoRA adapters, RAG indexes, and augmented datasets.

## Dataset format

The expected post-training format is one conversation per line in `jsonl`:

```json
{
  "id": "example-1",
  "messages": [
    {
      "role": "system",
      "content": "You are a virtual assistant from the Canary Islands. You answer using typical Canary vocabulary, syntax, and expressions."
    },
    {
      "role": "user",
      "content": "What is your favorite food?"
    },
    {
      "role": "assistant",
      "content": "Chacho, if you put a good plate in front of me..."
    }
  ],
  "metadata": {
    "split": "train"
  }
}
```

If `metadata.split` is missing, the scripts automatically assign examples to `train` and `validation`.

Another valid example, closer to a real multi-turn conversation:

```json
{
  "id": "example-2",
  "messages": [
    {
      "role": "system",
      "content": "You are 1xe, a virtual assistant from the Canary Islands. Keep a natural Canary tone, without caricature."
    },
    {
      "role": "user",
      "content": "I'm going to Las Palmas for a few days. What should I know before moving around by bus?"
    },
    {
      "role": "assistant",
      "content": "Here we usually call the bus the guagua. If you're moving around the city, check the urban lines first, and if you're going farther out, look at the intercity ones too."
    },
    {
      "role": "user",
      "content": "And if I ask someone on the street?"
    },
    {
      "role": "assistant",
      "content": "No problem at all. Ask naturally and they'll usually explain it to you just fine. If you say guagua instead of bus, you'll sound much closer to how people speak here."
    }
  ],
  "metadata": {
    "split": "train",
    "source": "canary_style_converation"
  }
}
```

## Installation

```bash
uv sync
```

Then you can verify the environment with:

```bash
uv run python --version
```

## Data preparation

Main scripts:

- [`siani/data_preparation/scrapper_main.py`](siani/data_preparation/scrapper_main.py)
- [`siani/data_preparation/scrapper.py`](siani/data_preparation/scrapper.py)
- [`siani/data_preparation/data_cleasing_main.py`](siani/data_preparation/data_cleasing_main.py)

This layer is used for scraping, normalization, and cleaning of the original corpus.

## Training 1: Canary style

Script:

- [`siani/post_training/train_qwen_style.py`](siani/post_training/train_qwen_style.py)

What it does:

- loads `Qwen/Qwen2.5-7B-Instruct`
- trains a style-focused LoRA adapter
- uses all `.jsonl` files in `siani/data/post/`
- saves the checkpoint to:
  - `outputs/qwen_canarian_style_lora`

Run:

```bash
python -m siani.post_training.train_qwen_style
```

## Test 1: Canary style

Script:

- [`siani/post_training/test_qwen_style.py`](siani/post_training/test_qwen_style.py)

What it does:

- loads the style LoRA
- injects a Canary-style system prompt by default
- keeps short conversation memory across turns

Run:

```bash
python -m siani.post_training.test_qwen_style
```

## Training 2: style + RAG

Script:

- [`siani/post_training/train_qwen_style_rag.py`](siani/post_training/train_qwen_style_rag.py)

What it does:

- uses `siani/data/post/canary_style.jsonl`
- builds a RAG index from:
  - `siani/data/academia_canaria/`
  - `siani/data/canariwiki/`
  - `siani/data/gevic/`
- augments each training question with retrieved context
- starts from the style LoRA if it already exists, to preserve the Canary voice
- saves the final LoRA to:
  - `outputs/qwen_canarian_posttrain_style_rag_lora`
- also saves the augmented dataset to:
  - `outputs/canary_style_posttrain_rag_augmented.jsonl`

Run:

```bash
python -m siani.post_training.train_qwen_style_rag
```

## Test 2: style + RAG

Script:

- [`siani/post_training/test_qwen_style_rag.py`](siani/post_training/test_qwen_style_rag.py)

What it does:

- loads the `style+rag` LoRA
- indexes `academia_canaria`, `canariwiki`, and `gevic`
- decides whether to trigger RAG based on prompt intent
- keeps short conversation memory across turns
- reuses the previous topic for follow-up prompts such as `tell me more`

Run:

```bash
python -m siani.post_training.test_qwen_style_rag
```

## Main outputs

Checkpoints:

- `outputs/qwen_canarian_style_lora`
- `outputs/qwen_canarian_posttrain_style_rag_lora`

Artifacts:

- `outputs/canary_style_posttrain_rag_augmented.jsonl`
- `outputs/qwen_style_rag.sqlite3`

## Recommended workflow

1. Place source corpora inside `siani/data/...`
2. Place `canary_style.jsonl` inside `siani/data/post/`
3. Train style:

```bash
python -m siani.post_training.train_qwen_style
```

4. Test style:

```bash
python -m siani.post_training.test_qwen_style
```

5. Train style + RAG:

```bash
python -m siani.post_training.train_qwen_style_rag
```

6. Test style + RAG:

```bash
python -m siani.post_training.test_qwen_style_rag
```

## Current limitations

- quality depends heavily on `canary_style.jsonl`
- if the style dataset is too repetitive, the model will recycle stock answers
- RAG activation is still heuristic
- the RAG index is local and optimized for iteration, not production

## Reasonable next steps

- clean duplicated or templated answers from the style dataset
- add more short, natural conversational examples
- improve the RAG router
- avoid rebuilding the full index on every launch
