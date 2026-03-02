"""
Model loading utilities.
"""

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load the M2M100 model and tokenizer.

    Args:
        model_name: HuggingFace model identifier or local path.
        device: Target device ('cuda' or 'cpu').

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading tokenizer from '{model_name}' ...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)

    print(f"Loading model from '{model_name}' ...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

    print(f"Model loaded on: {device}")
    return model, tokenizer
