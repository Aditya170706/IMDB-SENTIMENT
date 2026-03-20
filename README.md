# 🎬 Movie Review Sentiment Analysis

A deep learning project that classifies IMDB movie reviews as **Positive** or **Negative** using a Bidirectional LSTM model built with TensorFlow/Keras.

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── IMDB Dataset.csv          # Raw dataset (50,000 reviews) — do NOT commit to Git
├── modelmaking.py            # Data preprocessing + model training script
├── app.py                    # Streamlit web UI
├── sentiment_model.h5        # Trained Bidirectional LSTM (16 MB) — do NOT commit
├── tokenizer.pkl             # Fitted tokenizer (7.8 MB) — do NOT commit
├── config.json               # max_len and vocab_size settings
├── .gitignore                # Excludes large files from Git
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download large files manually

These files are too large for Git — download them separately and place in the root folder:

| File | Size | Where to get |
|------|------|--------------|
| `IMDB Dataset.csv` | ~64 MB | [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| `sentiment_model.h5` | ~16 MB | Train using `modelmaking.py` or download from Google Colab |
| `tokenizer.pkl` | ~7.8 MB | Generated automatically by `modelmaking.py` |

### 4. Train the model (skip if you already have .h5 file)

```bash
python modelmaking.py
```

This will:
- Load and clean `IMDB Dataset.csv`
- Tokenize the reviews
- Train the Bidirectional LSTM
- Save `sentiment_model.h5`, `tokenizer.pkl`, and `config.json`

### 5. Launch the web app

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 🧠 Model Architecture

```
Input: Movie review text
        ↓
Embedding Layer  (vocab_size → 128 dimensions)
        ↓
SpatialDropout1D (0.2)
        ↓
Bidirectional LSTM (64 units)   ← reads forward + backward
        ↓
Dropout (0.3)
        ↓
Dense (1 unit, sigmoid)         ← 0 = Negative, 1 = Positive
```

### Why Bidirectional LSTM?

A normal LSTM reads left → right only. Bidirectional reads both directions — so it understands how words like "not" affect the meaning of words that come after them.

```
"The movie was not good"
  Normal LSTM  →  reads only left to right
  BiLSTM       →  reads both ways, understands "not" + "good" together
```

---

## ⚙️ How It Works

### Text Preprocessing Pipeline

```
Raw review
    ↓
Remove special characters (keep only a-z A-Z spaces)
    ↓
Lowercase everything
    ↓
Remove stopwords (the, is, a, an ...)
    ↓
Tokenize (words → numbers)
    ↓
Pad sequences to fixed length (max_len)
    ↓
Feed into LSTM model
```

### config.json explained

```json
{
  "max_len": 300,
  "vocab_size": 10000
}
```

These values must be identical during both training and prediction. Never change them after training — the model depends on them.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~88–91% |
| Precision | ~89% |
| Recall | ~88% |
| Dataset | IMDB 50K reviews |
| Train/Test split | 80% / 20% |

---

## 🖥️ Web App (app.py)

The Streamlit app lets you type any movie review and get an instant prediction.

**Features:**
- Live sentiment prediction (Positive / Negative)
- Confidence score shown as percentage
- Visual confidence progress bar

**How to use:**
1. Run `streamlit run app.py`
2. Type or paste a movie review into the text box
3. Click **Analyse Sentiment**
4. See the result with confidence score

---

## 📦 Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.8.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Save the above as `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

---

## 🔑 Key Files Explained

| File | Purpose |
|------|---------|
| `modelmaking.py` | Full pipeline: loads CSV → cleans text → trains BiLSTM → saves all files |
| `app.py` | Streamlit UI — loads saved model and tokenizer, takes user input |
| `sentiment_model.h5` | Trained neural network weights — required by `app.py` |
| `tokenizer.pkl` | Word-to-number mapping — must match the one used during training |
| `config.json` | Stores `max_len` and `vocab_size` so `app.py` uses correct settings |
| `IMDB Dataset.csv` | 50,000 labelled movie reviews (positive/negative) from Kaggle |

---

## ⚠️ Important Notes

### Tokenizer and model must always match

```
Training:   "amazing" → tokenizer → 245 → model learns: 245 = positive word
Prediction: "amazing" → same tokenizer → 245 → model predicts correctly ✅

Prediction: "amazing" → different tokenizer → 891 → model is confused ❌
```

**Never retrain the tokenizer without also retraining the model.**

### Google Colab users

Files in `/content/` are deleted when your session ends. Always download after training:

```python
from google.colab import files
files.download('/content/sentiment_model.h5')
files.download('/content/tokenizer.pkl')
files.download('/content/config.json')
```

---

## 🙈 .gitignore

```gitignore
# Large files — too big for GitHub
IMDB Dataset.csv
sentiment_model.h5
tokenizer.pkl

# Python cache
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Virtual environment
venv/
.env
```

---

## 📚 References

- [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [TensorFlow Keras LSTM Docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Streamlit Documentation](https://docs.streamlit.io)
- [NLTK Stopwords](https://www.nltk.org/)

---

## 👤 Author

Made with ❤️ as a beginner deep learning project.
Feel free to fork, improve, and share!
