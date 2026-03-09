"""
Model loading utilities.
"""

from pathlib import Path

from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def _is_lora_adapter_dir(model_name: str) -> bool:
    """Detect whether a local directory contains PEFT adapter weights."""
    model_path = Path(model_name)
    return model_path.is_dir() and (model_path / "adapter_config.json").is_file()


def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load the M2M100 model and tokenizer.

    Args:
        model_name: HuggingFace model identifier or local path.
        device: Target device ('cuda' or 'cpu').

    Returns:
        Tuple of (model, tokenizer).
    """
    if _is_lora_adapter_dir(model_name):
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path
        print(f"Loading tokenizer from adapter dir '{model_name}' ...")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        print(f"Loading base model from '{base_model_name}' ...")
        base_model = M2M100ForConditionalGeneration.from_pretrained(base_model_name)
        print(f"Loading LoRA adapters from '{model_name}' ...")
        model = PeftModel.from_pretrained(base_model, model_name).to(device)
    else:
        print(f"Loading tokenizer from '{model_name}' ...")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)

        print(f"Loading model from '{model_name}' ...")
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

    print(f"Model loaded on: {device}")
    return model, tokenizer


def apply_lora_adapters(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Attach LoRA adapters for sequence-to-sequence language modeling."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
