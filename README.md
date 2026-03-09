# ES↔ZH Translator Local con Hugging Face

## Integrantes
- David Fernandez
- Dario Cristian Indries

## Descripción
Esta aplicación traduce entre español y chino de forma local (sin APIs externas), usando un modelo de Hugging Face adaptado con LoRA. El objetivo es ofrecer traducciones más consistentes en el dominio trabajado durante el proyecto y demostrar un flujo reproducible de entrenamiento + uso.

## Modelo Base
- Modelo: `facebook/m2m100_418M`
- Motivo de elección:
  - Soporta traducción multilingüe de forma nativa.
  - Tiene buen equilibrio entre calidad y viabilidad para trabajo local.
  - Se integra bien con `transformers` + `Trainer`.

## Técnica de Adaptación
- Opción elegida: **Fine-tuning (Option A) con LoRA**.
- Justificación:
  - El ajuste completo del modelo supera fácilmente la VRAM disponible en GPUs pequeñas.
  - LoRA reduce parámetros entrenables y consumo de memoria.
  - Permite entrenar localmente manteniendo buena calidad.
- Resumen técnico:
  - Se carga el modelo base.
  - Se inyectan adaptadores LoRA en capas objetivo.
  - Se entrena sobre pares ES-ZH y se guardan adaptadores en `mt_es_zh_lora/`.

## Dataset
- Fuente: `news_commentary` (`es-zh`) desde Hugging Face Datasets.
- Procesamiento:
  - Split train/validación.
  - Subconjunto para entrenamiento rápido local.
  - Tokenización con longitud máxima configurable.

## Instalación y Ejecución

```bash
git clone https://github.com/davidfvaquero/es-zh_translation.git
cd es-zh_translation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Entrenar adaptadores (Fine-tuning)

```bash
python src/train.py
```

Si hay fragmentación de memoria CUDA:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py
```

### Ejecutar aplicación (CLI interactiva)

```bash
python src/app.py
```

### Ejecutar interfaz web (Flask)

```bash
PYTHONPATH=src python -m es_zh_translation.web
```

## Ejemplo de Uso

Ejemplo 1:
- Entrada: `Me llamo David y soy de España.`
- Salida esperada: 我叫大卫，来自西班牙.

Ejemplo 2:
- Entrada: `我叫大卫，来自西班牙。`
- Salida esperada: Mi nombre es David, soy de España.

Ejemplo 3:
- Entrada: `Hola mundo`
- Salida esperada: 你好世界.

## Estructura Real del Repositorio

```text
es-zh_translation/
├── mt_es_zh_lora/                  # Adaptadores LoRA entrenados (checkpoints)
├── src/
│   ├── app.py
│   ├── train.py
│   ├── utils.py
│   └── es_zh_translation/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── web.py
│       ├── translate.py
│       └── templates/
│           └── index.html
├── requirements.txt
├── README.md
└── presentation/
```

## Referencias
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Hugging Face Datasets: https://huggingface.co/docs/datasets
- Modelo M2M100: https://huggingface.co/facebook/m2m100_418M
- PEFT / LoRA: https://huggingface.co/docs/peft

## Autoevaluación
- Lo más difícil:
  - Ajustar el entrenamiento para evitar errores de memoria CUDA en GPU limitada.
- Resultados:
  - Se consiguió entrenamiento local funcional con LoRA.
  - La app permite traducción ES↔ZH en local.
- Mejoras futuras:
  - Curar dataset de dominio propio (50-100 QA/domain pairs).
  - Añadir interfaz web (Gradio/FastAPI) y evaluación automática de calidad.
