"""
Fine-tuning entry point (Option A: LoRA fine-tuning).

Run:
    python src/train.py
"""

from es_zh_translation import config
from es_zh_translation.data import load_and_split_dataset, prepare_datasets
from es_zh_translation.model import apply_lora_adapters, load_model_and_tokenizer
from es_zh_translation.train import get_training_args, train_model


def main():
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    if config.USE_LORA:
        model = apply_lora_adapters(
            model,
            r=config.LORA_R,
            alpha=config.LORA_ALPHA,
            dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
        )

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
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        logging_steps=config.LOGGING_STEPS,
    )

    train_model(model, tokenizer, train_tok, val_tok, training_args, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()

