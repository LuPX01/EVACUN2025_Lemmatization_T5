# EVACUN2025_Lemmatization_T5
Lemmatization using T5 model for cuneiform languages.

This repository contains code for lemmatization of cuneiform languages (Akkadian and Sumerian) using the [ByT5](https://huggingface.co/docs/transformers/model_doc/byt5) architecture. The system is developed for the EvaCun 2025 Shared Task.

We provide two versions of the lemmatization model:

- **Raw Lemma**: Predicts lemma with sense number (e.g., `nadānu I`)
- **Generalized Lemma**: Predicts base lemma only (e.g., `nadānu`)

---

## 📁 Folder Structure

```
EVACUN2025_T5_Lemmatization/
├── Raw_Lemma/                 # Scripts for raw lemma lemmatization
├── Generalized_Lemma/        # Scripts for generalized lemma lemmatization
├── models/                   # Trained model files (excluding large weights)
├── requirements.txt
└── README.md
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

## 🛠️ Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model

```bash
cd Raw_Lemma
python ByT5_TrainAndEval_Raw.py
```

For generalized version:

```bash
cd ../Generalized_Lemma
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

tokenizer = ByT5Tokenizer.from_pretrained("your-username/ByT5_small_RawLemma")
model = T5ForConditionalGeneration.from_pretrained("your-username/ByT5_small_RawLemma")
```

---

## 📄 License

This project is released under the MIT License.

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact the author at:

📧 your.email@example.com
