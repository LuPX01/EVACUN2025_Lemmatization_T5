# EVACUN2025_Lemmatization_T5
Lemmatization using T5 model for cuneiform languages.

This repository contains code for lemmatization of cuneiform languages (Akkadian and Sumerian) using the [ByT5](https://huggingface.co/docs/transformers/model_doc/byt5) and [mT5](https://huggingface.co/docs/transformers/model_doc/mt5) architecture. The system is developed for the EvaCun 2025 Shared Task.

We provide two versions of the lemmatization model:

- **Raw Lemma**: Predicts lemma with sense number (e.g., `nadānu I`)
- **Generalized Lemma**: Predicts base lemma only (e.g., `nadānu`)

---

## 📁 Folder Structure

```
EVACUN2025_T5_Lemmatization/
├── ByT5/
│   ├── Raw_Lemma/             # Scripts for raw lemma lemmatization
│   └── Generalized_Lemma/     # Scripts for generalized lemma lemmatization
├── mT5/
│   ├── Raw_Lemma/             # Scripts for raw lemma lemmatization
│   └── Generalized_Lemma/     # Scripts for generalized lemma lemmatization
├── README.md
└── requirements.txt
```

---

## 📥 Input Data Format

The input data should be an Excel file (`.xlsx`) with the following two columns:

```
clean_value     lemma
i-na            ina I
tam-hu-uṣ       tamhuṣu I
kak-ku          kakku I
ina             ina I
me-lul-tu₄      mēlultu I
```

- `clean_value`: tokenized word (in transliteration)
- `lemma`: target lemma with or without sense number

---

## 🚀 Usage

Each subfolder contains three scripts:

- `TrainAndEval_*.py` – for training and evaluation
- `Eval_*.py` – for evaluating the model on a test set
- `Predict_INPUT_*.py` – for interactive inference

You can run them directly in PyCharm by opening the desired script and clicking "Run".

### 1. Train the model

```bash
python ByT5_TrainAndEval_Raw.py
```

For generalized version:

```bash
python ByT5_TrainAndEval_Generalized.py
```

### 2. Evaluate on test set

```bash
python ByT5_Eval_Raw.py
```

or

```bash
python ByT5_Eval_Generalized.py
```

### 3. Run interactive prediction

```bash
python ByT5_Predict_INPUT_Raw.py
```

---

## 🤗 Pretrained Model

The trained model will be available on Hugging Face:

```python
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

tokenizer = ByT5Tokenizer.from_pretrained("PXXL/EVACUN2025_ByT5_small_RawLemma")
model = T5ForConditionalGeneration.from_pretrained("PXXL/EVACUN2025_ByT5_small_RawLemma")
```

---

## 📄 License

This project is released under the MIT License.
