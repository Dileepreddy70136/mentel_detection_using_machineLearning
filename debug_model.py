"""
debug_model.py - Stand-alone model tester
Run this OUTSIDE Flask to verify model works correctly.
Usage:  python debug_model.py
"""

import os, re, json, pickle, warnings, sys
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -- Preprocessing (MUST match train_model.py + app.py) --------
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'never', 'very', 'too', 'but'
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# -- Load Model ------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
model      = pickle.load(open(os.path.join(BASE, 'models', 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE, 'models', 'vectorizer.pkl'), 'rb'))

LABELS = {0: 'Happy/Normal', 1: 'Stressed/Anxious', 2: 'Depressed/Sad'}

# -- Test Suite ------------------------------------------------
TESTS = [
    ("feeling sad and hopeless",              2),
    ("I am happy and excited about life",     0),
    ("stressed about my exams tomorrow",      1),
    ("nothing matters anymore",               2),
    ("great day, feeling so energized",       0),
    ("anxious about my future, can't sleep",  1),
    ("I feel broken and empty inside",        2),
    ("wonderful weekend with family",         0),
    ("overwhelmed by work pressure",          1),
]

print("=" * 70)
print("  MODEL DEBUGGING REPORT")
print("=" * 70)

# Check 1: Model metadata
print(f"\n[INFO] Model classes: {model.classes_}")
print(f"[INFO] Number of TF-IDF features: {len(vectorizer.get_feature_names_out())}")

# Check 2: Run all tests
passed = 0
failed = 0
print(f"\n{'STATUS':<8} {'PREDICTED':<20} {'EXPECTED':<20} {'CONF':<8} INPUT")
print("-" * 70)

for text, expected in TESTS:
    processed = preprocess(text)
    vec       = vectorizer.transform([processed])
    pred      = int(model.predict(vec)[0])
    probs     = model.predict_proba(vec)[0]
    conf      = probs[pred]

    ok = pred == expected
    if ok: passed += 1
    else:  failed += 1

    status = "PASS" if ok else "FAIL"
    print(f"  {status:<6} {LABELS[pred]:<20} {LABELS[expected]:<20} {conf:.0%}     {text[:40]}")

    # Check 3: Is the vector empty?
    if vec.nnz == 0:
        print(f"         WARNING: EMPTY VECTOR after preprocessing: '{processed}'")

    # Check 4: Show all probs for failures
    if not ok:
        prob_str = " | ".join(f"{LABELS[i]}: {p:.0%}" for i, p in enumerate(probs))
        print(f"         -> All Probabilities: {prob_str}")

print("-" * 70)
print(f"\nResults: {passed}/{len(TESTS)} passed, {failed} failed")

if failed == 0:
    print("[OK] All tests passed! Model classifies correctly.")
else:
    print("[WARN] Some tests failed - model may need more training data.")

# Check 5: Bias test
print(f"\nBias Check:")
predictions = []
for text, _ in TESTS:
    p = preprocess(text)
    v = vectorizer.transform([p])
    predictions.append(int(model.predict(v)[0]))

unique = set(predictions)
if len(unique) == 1:
    print(f"   BIASED! Model always predicts: {LABELS[list(unique)[0]]}")
    print(f"   Fix: Re-run 'python train_model.py' with class_weight='balanced'")
elif len(unique) == 2:
    print(f"   PARTIAL BIAS: Only predicts {len(unique)} classes: {[LABELS[u] for u in sorted(unique)]}")
else:
    print(f"   OK - Model predicts all 3 classes: {[LABELS[u] for u in sorted(unique)]}")
