"""
CLI entry point for the ES-ZH translation project.

Usage
-----
    python main.py train                                   # fine-tune the model
    python main.py translate "Me llamo David"               # ES → ZH
    python main.py translate "我叫大卫，来自西班牙"           # ZH → ES
    python main.py translate --model ./mt_es_zh_lora "Hola" # use fine-tuned model
"""

import argparse
import os

import config
from model import load_model_and_tokenizer
from data import load_and_split_dataset, prepare_datasets
from train import get_training_args, train_model
from translate import translate


def cmd_train(args):
    """Run the full training pipeline."""
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)

    train_ds, val_ds = load_and_split_dataset(
        config.DATASET_NAME,
        config.LANG_PAIR,
        config.TEST_SIZE,
        config.SEED,
    )

    train_tok, val_tok = prepare_datasets(
        train_ds,
        val_ds,
        tokenizer,
        config.TRAIN_SUBSET_SIZE,
        config.VAL_SUBSET_SIZE,
        config.MAX_LENGTH,
    )

    training_args = get_training_args(
        output_dir=config.OUTPUT_DIR,
        batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        fp16=config.FP16,
        logging_steps=config.LOGGING_STEPS,
    )

    train_model(model, tokenizer, train_tok, val_tok, training_args, config.OUTPUT_DIR)


def cmd_translate(args):
    """Translate a single piece of text."""
    # Use fine-tuned model if available, otherwise base model.
    model_path = args.model if args.model else config.MODEL_NAME
    if args.model is None and os.path.isdir(config.OUTPUT_DIR):
        model_path = config.OUTPUT_DIR
        print(f"Found fine-tuned model at '{config.OUTPUT_DIR}', using it.")

    model, tokenizer = load_model_and_tokenizer(model_path, config.DEVICE)

    result = translate(args.text, model, tokenizer, config.DEVICE, config.MAX_NEW_TOKENS)
    print(f"\n{'='*50}")
    print(f"Input:  {args.text}")
    print(f"Output: {result}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Spanish ↔ Chinese translation (M2M100)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ───────────────────────────────────────────────────────────────
    subparsers.add_parser("train", help="Fine-tune the model on news_commentary ES-ZH")

    # ── translate ───────────────────────────────────────────────────────────
    p_translate = subparsers.add_parser("translate", help="Translate a sentence")
    p_translate.add_argument("text", type=str, help="Text to translate")
    p_translate.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model (default: auto-detect fine-tuned or base model)",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "translate":
        cmd_translate(args)


if __name__ == "__main__":
    main()
