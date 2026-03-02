# Spanish ↔ Chinese Translation (M2M100)

Fine-tune and run bidirectional Spanish–Chinese translation using Facebook's `m2m100_418M` model.

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** Training benefits from a CUDA GPU. On CPU it will work but run slowly. Edit `src/es_zh_translation/config.py` to adjust hyperparameters.

## Usage

### Fine-tune the model

```bash
PYTHONPATH=src python -m es_zh_translation.cli train
```

This will:
1. Download the `news_commentary` ES-ZH parallel corpus
2. Preprocess and tokenize 5 000 training / 500 validation examples
3. Fine-tune for 3 epochs and save to `./mt_es_zh_lora/`

### Translate text

```bash
# Spanish → Chinese
PYTHONPATH=src python -m es_zh_translation.cli translate "Me llamo David y soy de España"

# Chinese → Spanish (auto-detected)
PYTHONPATH=src python -m es_zh_translation.cli translate "我叫大卫，来自西班牙"

# Use a specific model path
PYTHONPATH=src python -m es_zh_translation.cli translate --model ./mt_es_zh_lora "Hola mundo"
```

The direction is auto-detected: if the input contains Chinese characters it translates to Spanish, otherwise to Chinese.

### Run a simple web UI

```bash
PYTHONPATH=src python -m es_zh_translation.web
```

Then open `http://127.0.0.1:5000` in your browser.

## Project Structure

| Path | Description |
|---|---|
| `src/es_zh_translation/config.py` | Hyperparameters, paths, device selection |
| `src/es_zh_translation/model.py` | Load M2M100 model & tokenizer |
| `src/es_zh_translation/data.py` | Dataset loading, prompt template, tokenization |
| `src/es_zh_translation/train.py` | Training pipeline (HuggingFace Trainer) |
| `src/es_zh_translation/translate.py` | Bidirectional auto-detect translation |
| `src/es_zh_translation/cli.py` | CLI entry point (`train` / `translate`) |
| `src/es_zh_translation/web.py` | Flask web server for translation |
| `src/es_zh_translation/templates/index.html` | Web UI template |
| `requirements.txt` | Dependencies |

## Configuration

All settings live in `src/es_zh_translation/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `facebook/m2m100_418M` | Base model |
| `TRAIN_SUBSET_SIZE` | `5000` | Number of training examples |
| `VAL_SUBSET_SIZE` | `500` | Number of validation examples |
| `NUM_EPOCHS` | `3` | Training epochs |
| `BATCH_SIZE` | `4` | Per-device batch size |
| `LEARNING_RATE` | `2e-4` | Learning rate |
| `FP16` | `True` | Mixed precision (requires GPU) |
| `MAX_NEW_TOKENS` | `80` | Max tokens during translation |
