"""
Spam Email/SMS Classifier - Demo and Inference Script.

This script loads the trained model and TF-IDF vectorizer to classify new,
unseen messages. It displays the predicted label ('SPAM' or 'HAM') and the
model's prediction confidence score (probability).

Usage:
    python predict.py
"""

import os
import re
import string
import joblib

# Paths to trained model artifacts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "spam_classifier.joblib")
VECTORIZER_PATH = os.path.join(CURRENT_DIR, "models", "tfidf_vectorizer.joblib")


def clean_text(text: str) -> str:
    """
    Standard text cleaner consistent with training pipeline:
    - Lowercase text
    - Strip URLs and special characters
    - Remove punctuation
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
    text = re.sub(r"\b\d{5,}\b", " phone_number ", text)
    text = re.sub(r"[\$£€¥]\d+", " money_amount ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts():
    """Loads the serialized model and TF-IDF vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{os.path.join(CURRENT_DIR, 'models')}'.\n"
            "Please run 'python train.py' first to train and save the model!"
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def classify_message(text: str, model, vectorizer) -> dict:
    """
    Classifies a single text message.
    
    Returns:
        dict with:
            - 'original_text': Original message
            - 'predicted_label': 'SPAM' or 'HAM'
            - 'spam_probability': float (0.0 to 1.0)
            - 'ham_probability': float (0.0 to 1.0)
            - 'confidence': Percentage string
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    
    # Class prediction
    pred_idx = model.predict(features)[0]  # 0 for ham, 1 for spam
    label = "SPAM" if pred_idx == 1 else "HAM"

    # Probability estimation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        ham_prob = probs[0]
        spam_prob = probs[1]
        confidence = probs[pred_idx] * 100
    elif hasattr(model, "decision_function"):
        # Fallback for uncalibrated linear SVM
        score = model.decision_function(features)[0]
        spam_prob = 1.0 / (1.0 + pow(2.71828, -score))
        ham_prob = 1.0 - spam_prob
        confidence = (spam_prob if pred_idx == 1 else ham_prob) * 100
    else:
        ham_prob = 1.0 if pred_idx == 0 else 0.0
        spam_prob = 1.0 if pred_idx == 1 else 0.0
        confidence = 100.0

    return {
        "original_text": text,
        "predicted_label": label,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence": f"{confidence:.2f}%"
    }


def run_demo():
    """Runs inference on 5 diverse benchmark messages and prints a formatted summary."""
    print("=" * 80)
    print(" " * 22 + "SPAM DETECTION DEMO INFERENCE")
    print("=" * 80)

    try:
        model, vectorizer = load_artifacts()
        print("[+] Successfully loaded trained model and TF-IDF vectorizer.\n")
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        return

    # Benchmark test messages
    test_messages = [
        # 1. Obvious lottery spam
        "CONGRATULATIONS! You have been chosen to win a £1,000 cash prize or an iPad! Call 09061701461 to claim your reward.",
        
        # 2. Urgent phishing spam
        "URGENT: Your Wells Fargo bank account security access is expiring. Click http://bit.ly/secure-auth-30 to verify credentials.",
        
        # 3. Legitimate personal / student message
        "Hey! Are you still at the university library? Let me know if you want to grab lunch together.",
        
        # 4. Legitimate academic / formal message
        "Hi Professor, I have attached my term paper draft for the Machine Learning class. Looking forward to your feedback.",
        
        # 5. Promo spam with numeric code
        "Free entry in 2 a weekly competition! Text WIN to 80082 now to claim 500 bonus points. T&C apply."
    ]

    print("Classifying 5 sample test messages:\n" + "-" * 80)
    
    for i, msg in enumerate(test_messages, 1):
        res = classify_message(msg, model, vectorizer)
        status_color = "[SPAM]" if res["predicted_label"] == "SPAM" else "[HAM] "
        print(f"\nMessage #{i}:")
        print(f"  Text: \"{msg}\"")
        print(f"  Result:      {status_color} (Confidence: {res['confidence']})")
        print(f"  Probabilities: Ham: {res['ham_probability']*100:.1f}% | Spam: {res['spam_probability']*100:.1f}%")

    print("\n" + "=" * 80)
    print("Interactive Mode: Test your own message below (Press Enter with empty text to exit).")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nEnter custom message: ").strip()
            if not user_input:
                print("Exiting demo. Thank you!")
                break
            res = classify_message(user_input, model, vectorizer)
            print(f"-> Prediction: {res['predicted_label']} (Confidence: {res['confidence']})")
            print(f"   Breakdown: Ham={res['ham_probability']*100:.1f}%, Spam={res['spam_probability']*100:.1f}%")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    run_demo()

