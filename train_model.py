"""
train_model.py — Advanced Mental Health NLP Training Pipeline
─────────────────────────────────────────────────────────────
Improvements over basic version:
  • NLTK lemmatization + proper stopword removal
  • TF-IDF with bigrams (ngram_range=1,2)
  • Balanced class weights  ← kills Neutral bias
  • Stratified train-test split
  • Cross-validation with F1 weighted scoring
  • Full classification report (Precision / Recall / F1)
  • Sanity-check predictions printed before saving
  • Saves model metadata (labels, version) alongside .pkl
"""

import os
import re
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
import pickle

os.makedirs('models', exist_ok=True)

# ── NLTK Assets ───────────────────────────────────────────────
lemmatizer  = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'never', 'very', 'too', 'but'   # keep negations
}

# ─────────────────────────────────────────────────────────────
#  LABELS:  0 = Happy/Normal   1 = Stressed/Anxious  2 = Depressed/Sad
# ─────────────────────────────────────────────────────────────

RAW_DATA = [
    # ══ Label 0: Happy / Normal ══════════════════════════════
    ("I am feeling great today and very happy!", 0),
    ("The weather is beautiful and I love it.", 0),
    ("I succeeded in my project and feel proud.", 0),
    ("I had a wonderful day at the park.", 0),
    ("Feeling energized and full of life!", 0),
    ("I feel amazing and motivated right now.", 0),
    ("Everything is going really well for me.", 0),
    ("I enjoyed a great meal and feel content.", 0),
    ("I'm excited about my new opportunity!", 0),
    ("My friends made me feel so loved today.", 0),
    ("I completed my goals and I feel proud of myself.", 0),
    ("Today was a great day, feeling super positive.", 0),
    ("I feel joyful and at peace with everything.", 0),
    ("Life is good and I am grateful today.", 0),
    ("I just got some awesome news and I'm thrilled!", 0),
    ("Feeling strong and confident today.", 0),
    ("I woke up refreshed and ready for the day.", 0),
    ("I feel happy, healthy, and balanced.", 0),
    ("Things are improving and I feel hopeful.", 0),
    ("I'm smiling today, life feels meaningful.", 0),
    ("Had a productive session at the gym, feeling alive!", 0),
    ("My family dinner was wonderful and warm.", 0),
    ("Laughed so hard today with my best friends.", 0),
    ("Feeling blessed and grateful for what I have.", 0),
    ("I got a promotion! I am overjoyed.", 0),

    # ══ Label 1: Stressed / Anxious ══════════════════════════
    ("I'm so stressed about my upcoming exams.", 1),
    ("The workload is overwhelming and I can't cope.", 1),
    ("My heart is racing, I feel like I'm having a panic attack.", 1),
    ("I am worried about my future and financial status.", 1),
    ("Everything is moving too fast and I feel anxious.", 1),
    ("I can't sleep because I keep overthinking everything.", 1),
    ("Too many deadlines and I don't know where to start.", 1),
    ("I feel tense and irritable all the time lately.", 1),
    ("My mind won't stop racing with anxious thoughts.", 1),
    ("I feel so much pressure from work and family.", 1),
    ("I'm nervous and can't calm down.", 1),
    ("I feel panicked about the interview tomorrow.", 1),
    ("Anxiety is taking over my thoughts today.", 1),
    ("I'm scared of failing and letting everyone down.", 1),
    ("I have too much on my plate and feel overwhelmed.", 1),
    ("I feel restless and can't focus on anything.", 1),
    ("My chest feels tight because I'm so worried.", 1),
    ("Everything feels urgent and I'm running out of time.", 1),
    ("I feel stressed and burnt out from work.", 1),
    ("I'm worried sick and can't eat or sleep properly.", 1),
    ("The pressure from deadlines is crushing me.", 1),
    ("I feel like I'm going to break down any moment.", 1),
    ("Constant worry about things I cannot control.", 1),
    ("My anxiety won't let me relax even for a minute.", 1),
    ("I feel overwhelmed by all the responsibilities.", 1),

    # ══ Label 2: Depressed / Sad ══════════════════════════════
    ("I feel completely alone and hopeless.", 2),
    ("I have no energy to get out of bed, everything feels dark.", 2),
    ("I've been crying all day and don't know why.", 2),
    ("Nothing matters anymore, I feel empty inside.", 2),
    ("I wish I could just disappear, life feels too heavy.", 2),
    ("I feel sad and lost, nothing brings me joy.", 2),
    ("I'm feeling very low and don't see the point of anything.", 2),
    ("I feel worthless and like a burden to everyone.", 2),
    ("Everything feels meaningless and I don't care anymore.", 2),
    ("I am so deeply unhappy and I don't know why.", 2),
    ("feeling sad and hopeless today", 2),
    ("I feel down and can't shake this sadness.", 2),
    ("I've lost interest in everything I used to enjoy.", 2),
    ("Depression is crushing me and nothing feels real.", 2),
    ("I feel disconnected from the world and everyone in it.", 2),
    ("I just want to stay in bed and never get up.", 2),
    ("I feel broken inside and don't see a way out.", 2),
    ("Crying for no reason and feeling completely numb.", 2),
    ("I feel miserable and alone, no one understands me.", 2),
    ("Life feels pointless and I feel totally defeated.", 2),
    ("I am so sad and tired of pretending I'm okay.", 2),
    ("feeling blue and like nothing will ever get better", 2),
    ("I don't see any hope left for my future.", 2),
    ("The darkness inside me is getting worse every day.", 2),
    ("I'm exhausted from fighting my own thoughts every day.", 2),
]

# ── Text Preprocessing ────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Pipeline:
      1. Lowercase
      2. Remove URLs, mentions, special chars
      3. Tokenize
      4. Remove stopwords (but keep negations)
      5. Lemmatize
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Build DataFrame ───────────────────────────────────────────
df = pd.DataFrame(RAW_DATA, columns=['text', 'label'])
df['processed'] = df['text'].apply(preprocess)

print(f"Dataset: {len(df)} samples | {df['label'].value_counts().to_dict()}")

# ── Stratified Train / Test Split ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label'],
    test_size=0.20,
    random_state=42,
    stratify=df['label']    # ensures each split has all 3 classes
)

# ── Class Weights ─────────────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))
print(f"\nClass weights: {class_weight_dict}")

# ── TF-IDF Feature Extraction ─────────────────────────────────
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=8000,
    sublinear_tf=True,      # log normalization helps with long vs short text
    min_df=1,
    analyzer='word'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Model: Logistic Regression (Best for small NLP datasets) ──
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    multi_class='multinomial',
    C=2.0
)
model.fit(X_train_vec, y_train)

# ── Cross-Validation F1 Score ─────────────────────────────────
cv_scores = cross_val_score(
    LogisticRegression(max_iter=500, class_weight='balanced', C=2.0),
    vectorizer.transform(df['processed']),
    df['label'],
    cv=5,
    scoring='f1_weighted'
)
print(f"\n5-Fold Cross-Val F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Evaluation ────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
label_names = ['Happy/Normal', 'Stressed/Anxious', 'Depressed/Sad']

print("\n" + "="*55)
print("📊 MODEL EVALUATION — Test Set")
print("="*55)
print(classification_report(y_test, y_pred, target_names=label_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=label_names, columns=label_names))

# ── Sanity Checks ─────────────────────────────────────────────
SANITY_TESTS = [
    ("feeling sad and hopeless today",              2),
    ("I am very happy and excited today",           0),
    ("stressed about exam tomorrow, overwhelmed",   1),
    ("I feel broken and want to disappear",         2),
    ("great day, feeling energized",                0),
    ("anxious about my future, can't sleep",        1),
    ("I don't want to live anymore",                2),
    ("feeling really down lately",                  2),
    ("I enjoyed my weekend with family",             0),
]

label_name_map = {0: "Happy/Normal    ", 1: "Stressed/Anxious", 2: "Depressed/Sad   "}
print("\n🧪 Sanity Checks:")
print("-"*65)
all_correct = True
for text, expected_label in SANITY_TESTS:
    processed  = preprocess(text)
    vec        = vectorizer.transform([processed])
    pred       = int(model.predict(vec)[0])
    probs      = model.predict_proba(vec)[0]
    status     = "✅" if pred == expected_label else "❌"
    if pred != expected_label:
        all_correct = False
    print(f"  {status} [{label_name_map[pred]}] ({max(probs):.0%}) ← \"{text[:50]}\"")

print("-"*65)
print(f"Sanity check: {'All passed ✅' if all_correct else 'Some failed — consider expanding dataset'}")

# ── Save Artifacts ────────────────────────────────────────────
pickle.dump(model,      open('models/model.pkl',      'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

metadata = {
    "version":      "2.0",
    "labels":       {0: "Happy/Normal", 1: "Stressed/Anxious", 2: "Depressed/Sad"},
    "cv_f1_mean":   round(float(cv_scores.mean()), 4),
    "dataset_size": len(df),
    "features":     "TF-IDF bigrams, NLTK lemmatization, balanced class weights"
}
json.dump(metadata, open('models/metadata.json', 'w'), indent=2)

print(f"\n✅ Saved: models/model.pkl, models/vectorizer.pkl, models/metadata.json")
