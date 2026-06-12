from __future__ import annotations

from pathlib import Path

from siani.post_training.conversations.train import train_conversations
from siani.post_training.normal import train as normal_train


REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_canarian_conversations_on_normal_lora"


def main() -> None:
    print("[pipeline 1/2] Training the normal stage...")
    normal_train.main()

    print("[pipeline 2/2] Training the conversational stage on top of the normal checkpoint...")
    train_conversations(
        base_lora_checkpoint=normal_train.OUTPUT_DIR,
        output_dir=FINAL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
