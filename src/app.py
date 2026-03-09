"""
Main application entry point (interactive CLI translator).

Run:
    python src/app.py
"""

from es_zh_translation import config
from es_zh_translation.model import load_model_and_tokenizer
from es_zh_translation.translate import translate
from utils import resolve_model_path


def main():
    model_path = resolve_model_path(config.MODEL_NAME, config.OUTPUT_DIR)
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, config.DEVICE)

    print("\nSpanish <-> Chinese translator")
    print("Write a sentence and press Enter. Type 'exit' to quit.\n")

    while True:
        text = input("> ").strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        result = translate(text, model, tokenizer, config.DEVICE, config.MAX_NEW_TOKENS)
        print(result)


if __name__ == "__main__":
    main()

