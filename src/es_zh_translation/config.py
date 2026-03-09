"""
Configuration and constants for the ES-ZH translation project.
"""

import torch

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/m2m100_418M"

# ── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "news_commentary"
LANG_PAIR = "es-zh"
TEST_SIZE = 0.1
SEED = 42

# ── Preprocessing ───────────────────────────────────────────────────────────
TRAIN_SUBSET_SIZE = 5000
VAL_SUBSET_SIZE = 500
MAX_LENGTH = 96

# ── Training ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./mt_es_zh_lora"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
FP16 = True
GRADIENT_CHECKPOINTING = True
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LOGGING_STEPS = 50

# ── Translation ─────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 80
