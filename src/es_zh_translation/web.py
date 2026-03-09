"""
Simple Flask UI for ES ↔ ZH translation.
"""

from functools import lru_cache
import os

from flask import Flask, render_template, request

from . import config
from .model import load_model_and_tokenizer
from .translate import translate

app = Flask(__name__)


@lru_cache(maxsize=1)
def _load_runtime():
    model_path = config.MODEL_NAME
    if os.path.isdir(config.OUTPUT_DIR):
        model_path = config.OUTPUT_DIR
    model, tokenizer = load_model_and_tokenizer(model_path, config.DEVICE)
    return model, tokenizer


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    output = ""
    error = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            error = "Please enter text to translate."
        else:
            try:
                model, tokenizer = _load_runtime()
                output = translate(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    device=config.DEVICE,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                error = f"Translation failed: {exc}"

    return render_template(
        "index.html",
        text=text,
        output=output,
        error=error,
        device=config.DEVICE,
        model=config.MODEL_NAME,
        output_dir=config.OUTPUT_DIR,
    )


def main():
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
