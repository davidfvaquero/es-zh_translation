"""
Utility helpers for entrypoint scripts.
"""

from pathlib import Path


def resolve_model_path(default_model: str, output_dir: str) -> str:
    """
    Use the fine-tuned/adapted model directory if it exists, otherwise base model.
    """
    return output_dir if Path(output_dir).is_dir() else default_model

