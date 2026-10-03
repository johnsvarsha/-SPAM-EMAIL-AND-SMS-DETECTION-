"""
Dataset Loader for Spam Email/SMS Detection Project.

This module handles:
1. Downloading the classic SMS Spam Collection dataset from a public source.
2. Generating a realistic synthetic dataset (~1,000 messages) as a reliable offline fallback.
3. Saving and loading the dataset as a standardized CSV file ('data/spam_dataset.csv').
"""

import os
import random
import urllib.request
import pandas as pd

# Path configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(DATA_DIR, "spam_dataset.csv")

# Public raw dataset URLs
PRIMARY_DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
BACKUP_DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv"


def download_real_dataset() -> pd.DataFrame:
    """Attempts to download the real SMS Spam Collection dataset from online repositories."""
    print("[*] Attempting to download the SMS Spam Collection dataset...")
    
    # Try primary source (TSV format: label \t message)
    try:
        req = urllib.request.Request(
            PRIMARY_DATA_URL, 
            headers={"User-Agent": "Mozilla/5.0 (SpamDetectorEducationProject/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            df = pd.read_csv(response, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
            # Filter any invalid or empty rows
            df = df.dropna().drop_duplicates()
            df["label"] = df["label"].astype(str).str.lower().str.strip()
            df = df[df["label"].isin(["ham", "spam"])]
            print(f"[+] Successfully downloaded {len(df)} messages from primary repository.")
            return df
    except Exception as e:
        print(f"[-] Primary source failed ({e}). Trying backup repository...")

    # Try backup source
    try:
        req = urllib.request.Request(
            BACKUP_DATA_URL, 
            headers={"User-Agent": "Mozilla/5.0 (SpamDetectorEducationProject/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            df = pd.read_csv(response, encoding="latin-1")
            df = df.iloc[:, [0, 1]]
            df.columns = ["label", "message"]
            df = df.dropna().drop_duplicates()
            df["label"] = df["label"].astype(str).str.lower().str.strip()
            df = df[df["label"].isin(["ham", "spam"])]
            print(f"[+] Successfully downloaded {len(df)} messages from backup repository.")
            return df
    except Exception as e:
        print(f"[-] Backup download failed ({e}). Switching to synthetic generator...")

    return None


def generate_synthetic_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generates a realistic synthetic spam/ham dataset of ~1,000 messages.
    Used when no active internet connection is available.
    Maintains a natural ~85% ham / ~15% spam imbalance.
    """
    print(f"[*] Generating {num_samples} realistic synthetic spam and ham messages...")
    random.seed(42)

    # Realistic Ham message templates and variations
    ham_templates = [
        "Hey {name}, are we still meeting for lunch at {place} today?",
        "Can you send me the lecture notes for {subject}? I missed class this morning.",
        "Don't forget the assignment deadline is {day} midnight. Good luck!",
        "Thanks for your help with the {project} project, really appreciate it!",
        "Hi Mom, I will be home around {time}. Please save some dinner for me.",
        "Running about 10 minutes late due to traffic, see you soon!",
        "Did you review the draft proposal I emailed you earlier today?",
        "Hey, let me know when you are free for a quick call regarding our presentation.",
        "Happy Birthday {name}! Hope you have an awesome day ahead.",
        "The professor moved office hours to {day} at 3 PM in room 204.",
        "Let's catch up this weekend. Are you free on Saturday afternoon?",
        "Please find attached the updated project budget spreadsheet.",
        "Are you going to the university career fair tomorrow morning?",
        "Great job on the presentation today, everyone was very impressed!",
        "Can you pick up some milk and groceries on your way back?",
        "I just pushed the latest commit to our Git repository, check it out.",
        "Let's study together at the library around {time}.",
        "Hi Professor, could you clarify question 3 on the homework assignment?",
        "Your ride with Uber has arrived. Vehicle: Silver Toyota Camry.",
        "Here is the Wi-Fi password for the conference room: Welcome2026."
    ]

    # Realistic Spam message templates and variations
    spam_templates = [
        "CONGRATULATIONS! You have won a {prize}! Claim your prize now at {url} or call {number}. T&Cs apply.",
        "URGENT: Your bank account at {bank} has been temporarily suspended. Verify your identity immediately: {url}",
        "Free entry in 2 a weekly competition to win {prize}! Text WIN to {number}. Standard rates apply.",
        "ALERT: Your package delivery with tracking #{tracking} is on hold. Update your delivery address: {url}",
        "Exclusive offer! Get 0% APR on your credit card debt today. Call {number} now for pre-approval.",
        "You have (1) unread message from an admirer! Tap {url} to view photos now. Stop2optout.",
        "FINAL NOTICE: Your car warranty is about to expire. Renew today to avoid costly repairs: {url}",
        "Instant cash loan up to $5,000 approved! No credit check required. Apply in 2 mins: {url}",
        "WINNER! As a valued customer, you have been selected for a free $500 gift card. Claim now: {url}",
        "Crypto alert: Earn 300% weekly returns with our automated AI trading bot. Join VIP group: {url}",
        "IMPORTANT NOTICE: Security alert on your Netflix subscription. Update billing details to keep watching: {url}",
        "You have been awarded a complimentary 3-day luxury holiday resort stay. Reply YES to redeem or call {number}.",
        "HOT DEALS: Buy 1 get 2 free on all designer watches! Limited 24-hour flash sale: {url}",
        "Your tax refund of $1,420.50 is ready to deposit. Submit your routing info here: {url}",
        "Special bonus: 50 Free Spins waiting in your account! Play casino games now and cash out: {url}"
    ]

    names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Emma", "Daniel", "Sarah", "Varsha"]
    places = ["the cafeteria", "Starbucks", "the campus hub", "the dining hall", "downtown"]
    subjects = ["Machine Learning", "Linear Algebra", "Data Structures", "Operating Systems", "Physics"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    times = ["2:30 PM", "5:00 PM", "6:15 PM", "1:00 PM", "4:45 PM"]
    prizes = ["$1,000 Cash", "an iPhone 15 Pro", "a $500 Amazon Gift Card", "a luxury Caribbean Cruise"]
    urls = ["http://bit.ly/claim-prize-today", "http://tinyurl.com/secure-login-39", "http://bonus-reward24.com", "http://win-gift-now.net"]
    numbers = ["88088", "54321", "90123", "+18005550199", "80082"]
    banks = ["Chase", "Wells Fargo", "Bank of America", "Citibank"]
    trackings = ["US9823101", "PKG-44921", "FEDEX-88219", "DHL-9021"]
    projects = ["Capstone", "AI Term Paper", "Database Schema", "Software Lab"]

    records = []
    # 85% ham, 15% spam
    num_ham = int(num_samples * 0.85)
    num_spam = num_samples - num_ham

    for _ in range(num_ham):
        template = random.choice(ham_templates)
        msg = template.format(
            name=random.choice(names),
            place=random.choice(places),
            subject=random.choice(subjects),
            day=random.choice(days),
            time=random.choice(times),
            project=random.choice(projects)
        )
        records.append({"label": "ham", "message": msg})

    for _ in range(num_spam):
        template = random.choice(spam_templates)
        msg = template.format(
            prize=random.choice(prizes),
            url=random.choice(urls),
            number=random.choice(numbers),
            bank=random.choice(banks),
            tracking=random.choice(trackings)
        )
        records.append({"label": "spam", "message": msg})

    random.shuffle(records)
    df = pd.DataFrame(records)
    print(f"[+] Generated synthetic dataset: {len(df)} total ({num_ham} ham, {num_spam} spam).")
    return df


def load_or_create_dataset(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Loads dataset from local CSV if it exists; otherwise downloads the real dataset
    or generates a synthetic dataset fallback, then saves to 'data/spam_dataset.csv'.
    """
    if os.path.exists(DATASET_CSV):
        print(f"[*] Found cached dataset at: {DATASET_CSV}")
        df = pd.read_csv(DATASET_CSV)
        print(f"[+] Loaded {len(df)} messages from cache.")
        return df

    if not force_synthetic:
        df = download_real_dataset()
    else:
        df = None

    if df is None or len(df) == 0:
        print("[!] Using synthetic dataset generator.")
        df = generate_synthetic_dataset(num_samples=1000)

    # Save to local CSV for reproducible runs
    df.to_csv(DATASET_CSV, index=False)
    print(f"[+] Saved dataset to '{DATASET_CSV}' for future runs.\n")
    return df


if __name__ == "__main__":
    df = load_or_create_dataset()
    print("\nDataset Preview:")
    print(df.head())
    print("\nClass Distribution:")
    print(df["label"].value_counts())

