"""
Training pipeline for the ES-ZH translation model.
"""

from transformers import TrainingArguments, Trainer


def get_training_args(
    output_dir: str,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    fp16: bool = True,
    logging_steps: int = 50,
) -> TrainingArguments:
    """Build a TrainingArguments object with the given hyperparameters."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        report_to="none",
    )


def train_model(model, tokenizer, train_dataset, val_dataset, training_args, output_dir: str):
    """
    Train the model and save the result.

    Args:
        model: The M2M100 model.
        tokenizer: The corresponding tokenizer.
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset.
        training_args: A TrainingArguments instance.
        output_dir: Where to save the fine-tuned model.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving model to '{output_dir}' ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
