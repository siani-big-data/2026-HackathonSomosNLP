<div align="center">

# ⬜🟦🟨 1xe ⬜🟦🟨

### Canary-style Spanish assistant with post-training and RAG

**Made in the Canary Islands**

[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers 4.57+](https://img.shields.io/badge/Transformers-4.57%2B-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers/)
[![Qwen 2.5](https://img.shields.io/badge/Qwen-2.5--7B-5B6CFF?style=for-the-badge)](https://huggingface.co/Qwen)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-10B981?style=for-the-badge)](https://github.com/huggingface/peft)
[![RAG](https://img.shields.io/badge/RAG-local%20sqlite%20fts5-7C3AED?style=for-the-badge)](https://sqlite.org/fts5.html)
[![uv managed](https://img.shields.io/badge/uv-managed-222222?style=for-the-badge)](https://docs.astral.sh/uv/)
[![Status prototype](https://img.shields.io/badge/status-research%20prototype-8B5CF6?style=for-the-badge)](https://github.com/siani-big-data/2026-HackathonSomosNLP)

1xe is a research prototype for building a Canary Islands Spanish assistant that understands and uses Canary Spanish, with post-training over conversation datasets and optional RAG over curated cultural sources such as Academia Canaria, Canariwiki, and GEVIC.

</div>

## Overview

This repository explores how to adapt Qwen into a Canary Islands Spanish assistant through:

- post-training on Canary-style conversations
- optional retrieval over curated cultural and lexical sources
- local evaluation scripts for interactive testing

## Datasets

The repository uses two broad dataset families:

- Style datasets in `siani/data/post/`
  These are conversation datasets in `jsonl` format used to teach the model how to answer in natural Canary Spanish.

- Knowledge datasets in `siani/data/academia_canaria/`, `siani/data/canariwiki/`, and `siani/data/gevic/`
  These are reference corpora used by the RAG pipeline to retrieve factual and cultural context at inference time.

Each training example is stored as one JSON object per line and contains a `messages` field with chat turns such as `system`, `user`, and `assistant`.

Example:

```json
{
  "id": "example-1",
  "messages": [
    {
      "role": "system",
      "content": "You are a virtual assistant from the Canary Islands. Keep a natural Canary tone."
    },
    {
      "role": "user",
      "content": "How do people in Gran Canaria usually say bus?"
    },
    {
      "role": "assistant",
      "content": "Here people usually say guagua."
    }
  ],
  "metadata": {
    "split": "train"
  }
}
```

## Training

The training pipeline starts from Qwen and adapts it with post-training datasets written in Canary Spanish. The main goal is to preserve the base model's general capabilities while teaching it a natural Canary dialect style.

[![1xe Training](figures/1xe_training.png)](figures/1xe_training.pdf)

## Inference

At inference time, the model can run in plain conversational mode or with retrieval enabled. When RAG is active, the system looks up relevant context in curated Canary knowledge sources before generating the answer.

[![1xe Inference](figures/1xe_generation.png)](figures/1xe_generation.pdf)

## Authors

- [Óscar Rico Rodríguez](https://github.com/orr21)
- [Ricardo Cárdenes](https://github.com/ricardocardn)
- [José Juan Hernández Gálvez](https://github.com/josejuanhernandezgalvez)

## License and copyright

Copyright © 2026 Óscar Rico Rodríguez, Ricardo Cárdenes, and José Juan Hernández Gálvez.

This project is released under the [MIT License](LICENSE).
