# Spam Email & SMS Detection System (Machine Learning)

An end-to-end Natural Language Processing (NLP) and Machine Learning project built in Python to classify emails and SMS messages into **HAM** (legitimate) or **SPAM** (unwanted/phishing/promotional).

---

## 📌 Project Overview

Spam detection is one of the classic real-world applications of Natural Language Processing and Supervised Machine Learning. This project follows standard industry best practices:
1. **Automated Data Pipeline**: Automatically retrieves the standard SMS Spam Collection dataset (or generates an offline synthetic dataset fallback if internet is unavailable).
2. **Exploratory Data Analysis (EDA)**: Analyzes class imbalances, sample messages, and length discrepancies between legitimate and spam messages.
3. **Text Preprocessing**: Normalizes text by lowercasing, tokenizing, removing stopwords, cleaning punctuation, and mapping URLs/phone numbers/currency into semantic tokens.
4. **Feature Extraction**: Converts raw text into numerical representations via **TF-IDF (Term Frequency - Inverse Document Frequency)** with n-grams, and benchmarks against **Bag-of-Words (CountVectorizer)**.
5. **Stratified Splitting & 5-Fold Cross-Validation**: Uses 80/20 stratified train/test split to maintain class proportions, validated across 5 cross-validation folds to guarantee stability.
6. **Multi-Model Comparison**: Evaluates 3 standard classifiers:
   - **Multinomial Naive Bayes (`MultinomialNB`)**
   - **Logistic Regression (`LogisticRegression`)**
   - **Linear Support Vector Machine (`LinearSVC` with Calibrated Probabilities)**
7. **Model Interpretability**: Uncovers the top 20 keywords and n-grams most strongly associated with spam.
8. **Model Persistence**: Serializes the best-performing model and vectorizer with `joblib`.
9. **Interactive Demo Inference (`predict.py`)**: Demonstrates real-time classification with predicted class and calibrated confidence score.

---

## 🗂️ Project Directory Structure

```
ML model/
├── data/
│   ├── dataset_loader.py       # Handles downloading & fallback synthetic dataset generation
│   └── spam_dataset.csv        # Local cached dataset (ham & spam labeled text)
├── models/
│   ├── spam_classifier.joblib  # Serialized best trained model
│   └── tfidf_vectorizer.joblib # Serialized fitted TF-IDF vectorizer
├── train.py                    # Complete end-to-end training and evaluation pipeline
├── predict.py                  # Demo script for testing unseen emails with confidence scores
├── requirements.txt            # Python package dependencies
├── model_comparison.png        # High-resolution performance charts and confusion matrix
└── README.md                   # Academic project documentation
```

---

## 🎓 Academic Concepts to Explain to Your Professor

### 1. Why TF-IDF over Bag-of-Words?
- **Bag-of-Words (CountVectorizer)** merely counts the raw occurrence count of each word in a document. However, frequent words (like "the", "message", "today") may appear often without carrying much discriminative signal.
- **TF-IDF (Term Frequency - Inverse Document Frequency)** solves this:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
  Words that appear across almost all emails get a low IDF score, while distinctive spam words (like "claim", "won", "urgent", "prize") receive high weights.

### 2. Why Evaluate 3 Models?
- **Multinomial Naive Bayes**: A probabilistic baseline based on Bayes' Theorem with the assumption of conditional feature independence. Extremely fast and surprisingly effective for high-dimensional sparse text data.
- **Logistic Regression**: A linear model that estimates log-odds of a message being spam. Provides calibrated probabilities and interpretable feature coefficients.
- **Linear Support Vector Machine (LinearSVC)**: Finds the maximum-margin hyperplane separating the two classes in high-dimensional TF-IDF space. Robust against overfitting on sparse text data. Wrapped with `CalibratedClassifierCV` (Platt scaling) to output calibrated probabilities.

### 3. Why F1-Score & High Precision Matter (Trade-Off Analysis)
In spam detection, **not all errors are equal**:
- **False Positive (FP)**: A genuine email (e.g., job offer, exam schedule, password reset) is incorrectly classified as SPAM and hidden in the junk folder. This can cause critical damage to the user.
- **False Negative (FN)**: A spam message is incorrectly classified as HAM and lands in the inbox. This causes minor annoyance, but no critical information is lost.
- **Conclusion**: A production spam filter must achieve **near 100% Precision** while maintaining high Recall and F1-Score. Accuracy alone is deceiving because ham represents ~87% of all messages (a dummy model predicting 100% ham would achieve ~87% accuracy while failing completely at detecting spam).

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Complete Training Pipeline
```bash
python train.py
```
This will:
- Load the dataset and display descriptive EDA metrics.
- Preprocess messages and demonstrate the BoW vs. TF-IDF comparison.
- Train all 3 classifiers with 5-fold cross validation.
- Output detailed test set metrics (Accuracy, Precision, Recall, F1).
- Identify and print top spam keywords.
- Save the visual evaluation plot to `model_comparison.png`.
- Export the trained model and vectorizer into `models/`.

### 3. Run the Demo & Interactive Prediction
```bash
python predict.py
```
This will:
- Load the saved model and vectorizer.
- Classify 5 diverse benchmark messages with confidence percentages.
- Allow you to enter any custom message directly in the terminal to test live predictions.

---

## 📊 Summary of Evaluation Metrics

| Metric | Formula | Academic Meaning |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage of correct classifications |
| **Precision** | $\frac{TP}{TP + FP}$ | Out of all messages flagged as spam, how many were actually spam? (Minimizes false alarms) |
| **Recall** | $\frac{TP}{TP + FN}$ | Out of all actual spam messages, how many were successfully caught? |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall |

