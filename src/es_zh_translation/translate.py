"""
Bidirectional Spanish ↔ Chinese translation.
"""

import re
import unicodedata
import torch


def translate(text: str, model, tokenizer, device: str, max_new_tokens: int = 80) -> str:
    """
    Auto-detect direction and translate between Spanish and Chinese.

    If the input contains Chinese characters → translate to Spanish.
    Otherwise → translate to Chinese.

    Args:
        text: Input text to translate.
        model: A loaded M2M100 model.
        tokenizer: The corresponding M2M100 tokenizer.
        device: 'cuda' or 'cpu'.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Translated string.
    """
    contains_chinese = re.search(r'[\u4e00-\u9fff]', text) is not None

    if contains_chinese:
        tokenizer.src_lang = "zh"
        target_lang = "es"
    else:
        tokenizer.src_lang = "es"
        target_lang = "zh"

    encoded = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            max_new_tokens=max_new_tokens,
        )

    decoded = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    # Normalize compatibility glyphs and drop known malformed markers.
    cleaned = unicodedata.normalize("NFKC", decoded).replace("\ufffd", "").replace("\ufeff", "")
    return cleaned.strip()
