"""
================================================================================
Spam Email / SMS Detection Machine Learning Pipeline
================================================================================

This script implements an end-to-end Machine Learning NLP pipeline:
1. Dataset Loading & Exploratory Data Analysis (EDA)
2. Text Cleaning, Tokenization, and Stopword Filtering
3. Feature Extraction (TF-IDF Vectorization & Bag-of-Words Comparison)
4. Stratified Train/Test Splitting (80/20)
5. Model Training & Comparison:
   - Multinomial Naive Bayes (MultinomialNB)
   - Logistic Regression (LogisticRegression)
   - Linear Support Vector Machine (LinearSVC with Calibrated Probabilities)
6. 5-Fold Stratified Cross-Validation & Test Set Evaluation
7. Best Model Selection & Academic Explanation (Precision vs. Recall)
8. Top Spam Feature Extraction (Most Informative Words/Tokens)
9. Artifact Serialization (joblib) & Visual Evaluation Plot Generation

Author: Machine Learning Pair Programmer
Target Audience: Student / Academic Presentation
================================================================================
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import dataset loader
from data.dataset_loader import load_or_create_dataset

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOT_OUTPUT = os.path.join(BASE_DIR, "model_comparison.png")
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1: TEXT PREPROCESSING HELPERS
# ==============================================================================

# Standard English stopwords list (self-contained to avoid external NLTK download issues)
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until
up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves
""".split())


def clean_text(text: str) -> str:
    """
    Standard text normalization pipeline for email/SMS:
    1. Lowercases text
    2. Maps URLs, telephone numbers, and currency figures to special tokens
    3. Strips punctuation and digits
    4. Removes common english stopwords
    5. Normalizes redundant whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace special patterns with informative categorical tokens
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
    text = re.sub(r"\b\d{5,}\b", " phone_number ", text)
    text = re.sub(r"[\$£€¥]\d+", " money_amount ", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stopwords and short single-letter tokens
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    
    return " ".join(filtered_tokens)


# ==============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def perform_eda(df: pd.DataFrame):
    """Prints comprehensive descriptive statistics and sample messages."""
    print("\n" + "=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    total_samples = len(df)
    class_counts = df["label"].value_counts()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)
    
    print(f"Total Dataset Size: {total_samples:,} messages")
    print(f" - Ham  (Legitimate): {ham_count:,} ({ham_count / total_samples * 100:.2f}%)")
    print(f" - Spam (Unwanted)  : {spam_count:,} ({spam_count / total_samples * 100:.2f}%)")
    print("Note: Natural text datasets are imbalanced. Ham constitutes majority (~85-87%).\n")
    
    # Length metrics
    df["char_length"] = df["message"].astype(str).apply(len)
    df["word_count"] = df["message"].astype(str).apply(lambda s: len(s.split()))
    
    length_summary = df.groupby("label")[["char_length", "word_count"]].mean().reset_index()
    print("Average Message Length Statistics:")
    print("-" * 50)
    for _, row in length_summary.iterrows():
        print(f" Class: {row['label'].upper():<4} | Avg Characters: {row['char_length']:6.1f} | Avg Words: {row['word_count']:5.1f}")
    print("-" * 50)
    print("Observation: Spam messages are typically significantly longer and denser in words")
    print("than personal ham messages, often cramming promotional text and disclaimers.\n")
    
    # Display sample messages
    print("Sample Legitimate Messages (Ham):")
    for i, msg in enumerate(df[df["label"] == "ham"]["message"].head(2), 1):
        print(f" [{i}] \"{msg.strip()}\"")
        
    print("\nSample Spam Messages:")
    for i, msg in enumerate(df[df["label"] == "spam"]["message"].head(2), 1):
        print(f" [{i}] \"{msg.strip()}\"")
    print("=" * 80 + "\n")


# ==============================================================================
# SECTION 3: FEATURE EXTRACTION & COMPARISON (BoW vs. TF-IDF)
# ==============================================================================

def compare_vectorizers(X_train_raw, y_train, X_test_raw, y_test):
    """
    Demonstrates the difference between Bag-of-Words (CountVectorizer)
    and Term Frequency - Inverse Document Frequency (TfidfVectorizer).
    """
    print("=" * 80)
    print("STEP 2: FEATURE EXTRACTION COMPARISON (Bag-of-Words vs. TF-IDF)")
    print("=" * 80)
    
    # 1. Bag of Words
    bow_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=3000)
    X_train_bow = bow_vectorizer.fit_transform(X_train_raw)
    X_test_bow = bow_vectorizer.transform(X_test_raw)
    
    clf_bow = MultinomialNB()
    clf_bow.fit(X_train_bow, y_train)
    bow_pred = clf_bow.predict(X_test_bow)
    bow_f1 = f1_score(y_test, bow_pred)
    bow_acc = accuracy_score(y_test, bow_pred)
    
    # 2. TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    
    clf_tfidf = MultinomialNB()
    clf_tfidf.fit(X_train_tfidf, y_train)
    tfidf_pred = clf_tfidf.predict(X_test_tfidf)
    tfidf_f1 = f1_score(y_test, tfidf_pred)
    tfidf_acc = accuracy_score(y_test, tfidf_pred)
    
    print(f"Bag-of-Words  (CountVectorizer) -> Accuracy: {bow_acc:.4f} | F1-Score: {bow_f1:.4f}")
    print(f"TF-IDF        (TfidfVectorizer) -> Accuracy: {tfidf_acc:.4f} | F1-Score: {tfidf_f1:.4f}")
    print("Explanation: TF-IDF reduces the weight of words that occur across almost all documents")
    print("and boosts words that are distinctively frequent in specific spam contexts.\n")
    
    return tfidf_vectorizer, X_train_tfidf, X_test_tfidf


# ==============================================================================
# SECTION 4: MODEL TRAINING, 5-FOLD CV, AND EVALUATION
# ==============================================================================

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains 3 models, runs 5-fold cross-validation, evaluates on test set,
    and returns comprehensive metrics dictionary.
    """
    print("=" * 80)
    print("STEP 3 & 4: MODEL TRAINING & 5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, random_state=42),
        # LinearSVC wrapped in CalibratedClassifierCV yields calibrated probabilities for confidence scores
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42),
            method="sigmoid",
            cv=3
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n[*] Training & Validating '{name}'...")
        
        # 5-Fold Stratified Cross-Validation on Training Set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        print(f"    5-Fold CV F1-Scores: {[round(s, 4) for s in cv_scores]}")
        print(f"    Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Fit on full training set
        model.fit(X_train, y_train)
        
        # Predict on Test Set (80/20 held-out)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "cv_mean_f1": cv_scores.mean(),
            "cv_std_f1": cv_scores.std(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "y_pred": y_pred
        }
        
        print(f"    Test Accuracy : {acc:.4f}")
        print(f"    Test Precision: {prec:.4f}")
        print(f"    Test Recall   : {rec:.4f}")
        print(f"    Test F1-Score : {f1:.4f}")

    return results


# ==============================================================================
# SECTION 5: TOP SPAM FEATURES (EXPLAINABILITY)
# ==============================================================================

def display_top_spam_features(model, vectorizer, top_n=20):
    """
    Extracts and prints the top words/tokens most strongly associated with spam.
    """
    print("\n" + "=" * 80)
    print("STEP 5: TOP TF-IDF WORDS/FEATURES MOST ASSOCIATED WITH SPAM")
    print("=" * 80)
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Calculate feature importances based on model type
    if isinstance(model, LogisticRegression):
        coefficients = model.coef_[0]
        top_indices = np.argsort(coefficients)[::-1][:top_n]
        scores = coefficients[top_indices]
    elif isinstance(model, MultinomialNB):
        # Difference in log probability: log(P(w|spam)) - log(P(w|ham))
        log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_indices = np.argsort(log_prob_diff)[::-1][:top_n]
        scores = log_prob_diff[top_indices]
    elif isinstance(model, CalibratedClassifierCV):
        # Average coefficients across calibrated sub-estimators
        weights = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
        top_indices = np.argsort(weights)[::-1][:top_n]
        scores = weights[top_indices]
    else:
        print("[!] Model architecture does not expose direct linear feature weights.")
        return

    print(f"{'Rank':<5} | {'Spam Keyword / Feature':<25} | {'Importance Weight':<15}")
    print("-" * 52)
    for rank, (idx, score) in enumerate(zip(top_indices, scores), 1):
        word = feature_names[idx]
        print(f"{rank:<5} | {word:<25} | {score:+.4f}")
    print("-" * 52)
    print("Key Takeaway: Prompts with monetary indicators, urgency, links, and lottery")
    print("terms have strongly positive coefficients towards the 'SPAM' class.\n")


# ==============================================================================
# SECTION 6: VISUALIZATION (COMPARISON CHART & CONFUSION MATRIX)
# ==============================================================================

def generate_visual_plots(results: dict, best_model_name: str, y_test):
    """
    Generates and saves a clean, presentation-ready 2-panel figure:
    1. Multi-metric performance comparison bar chart
    2. Confusion matrix heatmap for the best model
    """
    print("=" * 80)
    print("STEP 6: GENERATING EVALUATION CHARTS")
    print("=" * 80)
    
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- PANEL 1: Model Comparison Bar Chart ---
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(model_names))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    
    for i, (m_key, m_label, col) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results[m][m_key] for m in model_names]
        bars = axes[0].bar(x + (i - 1.5) * width, vals, width, label=m_label, color=col, alpha=0.9)
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f"{height:.2f}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, rotation=0)
            
    axes[0].set_title("Model Comparison on Test Set (80/20 Split)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=10, fontweight="semibold")
    axes[0].set_ylim(0.80, 1.05)
    axes[0].legend(loc="lower right", frameon=True)
    
    # --- PANEL 2: Confusion Matrix Heatmap for Best Model ---
    cm = results[best_model_name]["confusion_matrix"]
    cm_labels = ["Ham (Legit)", "Spam"]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        ax=axes[1],
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    axes[1].set_title(f"Confusion Matrix: Best Model ({best_model_name})", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    axes[1].set_ylabel("True Label", fontsize=11, fontweight="semibold")
    
    # Annotate metrics inside confusion matrix quadrant explanation
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.78, 0.02, f"True Ham (TN): {tn} | False Spam (FP): {fp}\nMissed Spam (FN): {fn} | Caught Spam (TP): {tp}",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f3f4", edgecolor="#bdc3c7"))
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved evaluation figure to: {PLOT_OUTPUT}\n")


# ==============================================================================
# SECTION 7: MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print(" " * 20 + "STARTING SPAM DETECTION ML PIPELINE")
    print("=" * 80)
    
    # 1. Load Dataset
    df = load_or_create_dataset()
    
    # 2. Perform Exploratory Data Analysis
    perform_eda(df)
    
    # 3. Clean Text
    print("[*] Preprocessing message text (lowercase, punctuation, regex, stopwords)...")
    df["cleaned_message"] = df["message"].apply(clean_text)
    
    # Encode label: 'ham' -> 0, 'spam' -> 1
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    
    # 4. Stratified Train / Test Split (80% Train, 20% Test)
    X = df["cleaned_message"]
    y = df["label_num"]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"[+] Train set size: {len(X_train_raw):,} samples | Test set size: {len(X_test_raw):,} samples.")
    
    # 5. Feature Extraction Comparison (BoW vs TF-IDF)
    vectorizer, X_train_tfidf, X_test_tfidf = compare_vectorizers(
        X_train_raw, y_train, X_test_raw, y_test
    )
    
    # 6. Train Models & Evaluate with 5-Fold Cross-Validation
    results = train_and_evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # 7. Select Best Model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_res = results[best_model_name]
    best_model = best_res["model"]
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL SELECTED: {best_model_name.upper()}")
    print("=" * 80)
    print(f"  Test Accuracy : {best_res['accuracy'] * 100:.2f}%")
    print(f"  Test Precision: {best_res['precision'] * 100:.2f}%")
    print(f"  Test Recall   : {best_res['recall'] * 100:.2f}%")
    print(f"  Test F1-Score : {best_res['f1'] * 100:.2f}%")
    print(f"  5-Fold CV F1  : {best_res['cv_mean_f1'] * 100:.2f}% (std: {best_res['cv_std_f1'] * 100:.2f}%)")
    print("\n--- ACADEMIC EXPLANATION FOR MODEL SELECTION ---")
    print("Why F1-Score and High Precision matter in Spam Classification:")
    print("1. Severe Cost of False Positives: If a legitimate email (ham) is falsely classified")
    print("   as spam, critical messages (e.g. university admissions, banking alerts, family news)")
    print("   are sent to the junk folder where the user may never see them.")
    print("2. Mild Cost of False Negatives: If a spam message reaches the inbox, it causes minor")
    print("   annoyance to the user, but does not lose important correspondence.")
    print("3. Thus, while Accuracy can be misleading due to class imbalance (~87% ham), F1-Score")
    print("   harmonic mean balances Precision and Recall, and a high Precision guarantees")
    print("   very few false alarms on authentic communications.")
    
    # 8. Show Top Spam Features
    display_top_spam_features(best_model, vectorizer, top_n=20)
    
    # 9. Generate and Save Plots
    generate_visual_plots(results, best_model_name, y_test)
    
    # 10. Save Model and Vectorizer to Disk
    model_save_path = os.path.join(MODELS_DIR, "spam_classifier.joblib")
    vectorizer_save_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)
    print(f"[+] Model saved successfully to: {model_save_path}")
    print(f"[+] Vectorizer saved successfully to: {vectorizer_save_path}")
    
    # 11. Final Summary Table
    print("\n" + "=" * 80)
    print(" " * 28 + "FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Model Name':<26} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 74)
    for name, r in results.items():
        flag = " (Best)" if name == best_model_name else ""
        print(f"{name + flag:<26} | {r['accuracy']*100:6.2f}%   | {r['precision']*100:6.2f}%   | {r['recall']*100:6.2f}%   | {r['f1']*100:6.2f}%")
    print("=" * 80)
    print("\nPipeline execution complete! You can now run 'python predict.py' to test messages.\n")


if __name__ == "__main__":
    main()

