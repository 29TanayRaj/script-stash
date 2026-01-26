# Comprehensive Implementation Guide: Legal Clause Risk Classification

This document serves as the **master blueprint** for building a production-ready legal clause risk classifier. It includes theory, full code specifications, and operational details.

---

## 1. System Architecture

The system follows a classic **ETL (Extract, Transform, Load) + ML pipeline** architecture:

- **Data Ingestion Layer**: Automates downloading and caching of LEDGAR (training) and ContractNLI (validation) datasets.
- **Preprocessing Layer**: Normalizes legal text, handles encoding issues, and maps diverse specific legal topics to our three risk tiers.
- **Feature Engineering Layer**: Converts text to numerical vectors using TF-IDF with n-grams to capture multi-word legal terms of art (e.g., *"material breach"*).
- **Modeling Layer**: Trains probabilistic classifiers (Logistic Regression/SVM) with class weighting to handle imbalanced risk categories.
- **Inference Layer**: A lightweight API-ready function to predict risk and confidence scores for new clauses.

---

## 2. Environment Setup

### 2.1 Virtual Environment

Isolate dependencies to avoid conflicts.

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Dependencies (`requirements.txt`)

Exact versions for reproducibility.

```txt
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
nltk==3.8.1
datasets==2.14.4  # For downloading from Hugging Face
joblib==1.3.1     # For model serialization
tqdm==4.65.0      # Progress bars
pytest==7.4.0     # Testing
```

---

## 3. Data Strategy & Label Mapping

### 3.1 The Challenge

The LEDGAR dataset classifies clauses by **topic** (e.g., *"Governing Law"*, *"Indemnifications"*), not by risk. We must construct a ground truth mapping.

### 3.2 Risk Taxonomy (The "Gold Standard")

We will implement this mapping dictionary in `src/preprocess.py`.

| Risk Level     | Definition                                                   | Representative LEDGAR Labels |
|----------------|--------------------------------------------------------------|------------------------------|
| **Safe**       | Standard boilerplate with low liability impact.              | Governing Laws, Notices, Counterparts, Severability, Entire Agreements |
| **Needs Review** | Operational clauses that require business context.         | Term, Termination, Confidentiality, Assignment, Force Majeure |
| **High Risk**  | Clauses that directly impact financial liability or restrictions. | Indemnifications, Limitation of Liability, Warranties, Non-Competes, Liquidated Damages |

> **Note:** Any label not explicitly mapped will be dropped or marked as `Unclassified` during training to reduce noise.

---

## 4. Module Class Specifications

### 4.1 Data Ingestion (`src/download.py`)

**Responsibility**: Fetch data from Hugging Face and save raw artifacts.

```python
import os
import pandas as pd
from datasets import load_dataset

def download_ledgar(output_dir="data/raw"):
    """Downloads LEDGAR dataset from LexGLUE benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading LEDGAR dataset...")
    # 'ledgar' is a subset of the 'legal_glue' benchmark
    try:
        dataset = load_dataset("coastalcph/lex_glue", "ledgar")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Convert to pandas for easier local manipulation
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Validation set is sometimes 'validation' or part of train; check dataset keys
    if 'validation' in dataset:
         val_df = pd.DataFrame(dataset['validation'])
    else:
         val_df = pd.DataFrame()  # Handle empty

    # Save
    train_df.to_csv(os.path.join(output_dir, "ledgar_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "ledgar_test.csv"), index=False)
    if not val_df.empty:
        val_df.to_csv(os.path.join(output_dir, "ledgar_val.csv"), index=False)
        
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    download_ledgar()
```

---

### 4.2 Preprocessing Logic (`src/preprocess.py`)

**Responsibility**: Clean text and apply the risk taxonomy.

```python
import re
import pandas as pd

# Define the mapping explicitly
LABEL_MAPPING = {
    # SAFE
    "Governing Law": "Safe",
    "Notices": "Safe",
    "Counterparts": "Safe",
    "Severability": "Safe",
    "Entire Agreement": "Safe",
    
    # NEEDS REVIEW
    "Term": "Needs Review",
    "Termination": "Needs Review",
    "Confidentiality": "Needs Review",
    "Assignment": "Needs Review",
    
    # HIGH RISK
    "Indemnification": "High Risk",
    "Limitation of Liability": "High Risk",
    "Warranties": "High Risk",
    "Non-Compete": "High Risk"
}


def clean_text(text):
    """
    1. Lowercase
    2. Remove statutes/citations (optional, can be noise)
    3. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Multiline to single line
    text = text.strip()
    return text


def process_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # 1. Map Labels
    # LEDGAR label column is usually 'label' (int) or 'label_text' (str).
    # We assume we map from text labels. If int, we need the id-to-label map.
    # For this guide, assuming we have the text label class.
    
    # Filter only rows effectively mapped
    df['risk_label'] = df['label'].map(LABEL_MAPPING)
    df = df.dropna(subset=['risk_label'])
    
    # 2. Clean Text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 3. Save
    df[['clean_text', 'risk_label']].to_csv(output_path, index=False)
    print(f"Processed {len(df)} rows to {output_path}")
```

---

### 4.3 Feature Engineering & Training (`src/train.py`)

**Responsibility**: Train the TF-IDF Vectorizer and Classifier.

> **Key Technical Choice**: `TfidfVectorizer(ngram_range=(1,2))` captures phrases like *"shall be liable"* vs just *"liable"*.

```python
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(train_path, model_out_path):
    print("Loading data...")
    df = pd.read_csv(train_path)
    
    X = df['clean_text']
    y = df['risk_label']
    
    # Pipeline: Raw Text -> Vector -> Probability
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams and Bigrams
            max_features=20000,  # Cap features to prevent massive models
            stop_words='english',
            min_df=5             # Ignore terms appearing < 5 times
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',  # Crucial for imbalanced data
            solver='liblinear',        # Good for high-dimensional text
            C=1.0,                     # Regularization strength
            max_iter=1000
        ))
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    
    print("Saving artifact...")
    joblib.dump(pipeline, model_out_path)
    print(f"Model saved to {model_out_path}")
    
    return pipeline


if __name__ == "__main__":
    train_model("data/processed/train.csv", "models/risk_classifier.pkl")
```

---

### 4.4 Inference Engine (`src/predict.py`)

**Responsibility**: Load model once, predict many times.

```python
import joblib
import sys

class RiskClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.classes = self.model.classes_
        
    def predict(self, text):
        # The pipeline handles text preprocessing/vectorization automatically!
        pred_label = self.model.predict([text])[0]
        pred_probs = self.model.predict_proba([text])[0]
        
        # Get probability of the predicted class
        conf = max(pred_probs)
        
        return {
            "text_snippet": text[:50] + "...",
            "prediction": pred_label,
            "confidence": round(conf, 4),
            "all_scores": dict(zip(self.classes, pred_probs))
        }


if __name__ == "__main__":
    # simple CLI usage
    # python -m src.predict "User agrees to indemnify company..."
    classifier = RiskClassifier("models/risk_classifier.pkl")
    input_text = sys.argv[1] if len(sys.argv) > 1 else "This agreement shall be governed by the laws of California."
    
    result = classifier.predict(input_text)
    print(result)
```

---

## 5. Execution Roadmap

Follow these commands to build the system from scratch:

### Initialize Project

```bash
mkdir data models src tests
# (Create the files with the code above)
```

### Download Data

```bash
python -m src.download
```

### Process Data

> You’ll need to adapt `src/preprocess.py` to point to the downloaded raw files.

```bash
python -m src.preprocess
```

### Train

```bash
python -m src.train
```

### Test

```bash
python -m src.predict "User shall indemnify and hold harmless the Company."
# Expected: High Risk
```

---

## 6. Future Improvements (Post-MVP)

- **Transformer Upgrade**: Replace TF-IDF/LogisticRegression with a fine-tuned LegalBERT for better context understanding (requires GPU).
- **Explainability**: Use LIME or SHAP to highlight exactly which words triggered the *High Risk* label (lawyers love this).
- **Thresholding**: Instead of hard classes, return a *Review Priority Score* (0–100).
- **Clause Segmentation Model**: Currently we assume inputs are single clauses. In reality, you need a pre-step to split a full contract PDF into clauses.

