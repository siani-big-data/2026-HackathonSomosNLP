from __future__ import annotations

from pathlib import Path

from siani.post_training.conversations_rag.train import train_conversations_rag
from siani.post_training.normal import train as normal_train


REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_conversations_rag_on_normal_lora"
FINAL_AUGMENTED_DATASET_PATH = REPO_ROOT / "outputs" / "canary_style_conversation_rag_on_normal_augmented.jsonl"


def main() -> None:
    print("[pipeline 1/2] Training the normal stage...")
    normal_train.main()

    print("[pipeline 2/2] Training the conversations_rag stage on top of the normal checkpoint...")
    train_conversations_rag(
        base_lora_checkpoint=normal_train.OUTPUT_DIR,
        output_dir=FINAL_OUTPUT_DIR,
        augmented_dataset_path=FINAL_AUGMENTED_DATASET_PATH,
    )


if __name__ == "__main__":
    main()
