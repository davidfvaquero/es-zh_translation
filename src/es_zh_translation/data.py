"""
Dataset loading and preprocessing for ES-ZH translation.
"""

from datasets import load_dataset


def load_and_split_dataset(name: str, lang_pair: str, test_size: float, seed: int):
    """
    Load a parallel corpus and split it into train / validation sets.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    dataset = load_dataset(name, lang_pair)

    # The dataset only ships a "train" split, so we split manually.
    split = dataset["train"].train_test_split(test_size=test_size, seed=seed)

    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    return train_dataset, val_dataset


def _make_preprocess_fn(tokenizer, max_length: int):
    """Return a preprocessing function closed over the tokenizer."""

    def preprocess(example):
        es_text = example["translation"]["es"]
        zh_text = example["translation"]["zh"]

        prompt = (
            "You are a professional translation system.\n"
            "Translate the following sentence from Spanish to Chinese.\n"
            "Only output the translation. Do not explain anything.\n\n"
            f"Spanish: {es_text}\n"
            "Chinese:"
        )

        full_text = prompt + " " + zh_text

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return preprocess


def prepare_datasets(train_ds, val_ds, tokenizer, train_size: int, val_size: int, max_length: int):
    """
    Subset and tokenize the datasets.

    Returns:
        Tuple of (train_tokenized, val_tokenized).
    """
    train_ds = train_ds.select(range(min(train_size, len(train_ds))))
    val_ds = val_ds.select(range(min(val_size, len(val_ds))))

    preprocess = _make_preprocess_fn(tokenizer, max_length)

    train_tokenized = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    print(f"Tokenized train: {len(train_tokenized)}, val: {len(val_tokenized)}")
    return train_tokenized, val_tokenized
