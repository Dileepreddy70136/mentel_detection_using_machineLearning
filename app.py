"""
app.py — Production-ready Flask Mental Health Detection Backend
───────────────────────────────────────────────────────────────
Features:
  • SQLite database for persistent mood history (no data lost on restart)
  • Model loaded once at startup (not per-request)
  • Same NLTK preprocessing as training pipeline
  • Confidence-scaled wellness score
  • Proper error logging
  • /api/history endpoint for real-time graph updates
  • /api/stats endpoint for dashboard summary
"""

import os
import re
import json
import pickle
import sqlite3
import logging
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
from journal_generator import generate_journal

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), 'data', 'history.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
VEC_PATH   = os.path.join(os.path.dirname(__file__), 'models', 'vectorizer.pkl')
META_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'metadata.json')

# ── NLTK Setup (must match train_model.py exactly!) ────────────
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'never', 'very', 'too', 'but'
}

# ── Label Map ──────────────────────────────────────────────────
LABEL_MAP = {
    0: {
        "prediction":  "Healthy / Happy",
        "badge":       "success",
        "base_score":  82,
        "suggestions": [
            "Keep maintaining your positive routine! 🌟",
            "Share your positivity with someone today.",
            "Journal this mood to reflect on later.",
        ]
    },
    1: {
        "prediction":  "Stressed / Anxious",
        "badge":       "warning",
        "base_score":  38,
        "suggestions": [
            "Try the 4-7-8 breathing technique right now.",
            "Take a 10-minute break from screens.",
            "Talk to someone you trust about what's worrying you.",
            "iCall Helpline (India): 9152987821",
        ]
    },
    2: {
        "prediction":  "Depressed / Sad",
        "badge":       "danger",
        "base_score":  14,
        "suggestions": [
            "You are not alone — please reach out to someone. 💙",
            "iCall Helpline (India): 9152987821",
            "Vandrevala Foundation: 1860-2662-345 (24×7)",
            "Consider speaking with a mental health professional.",
            "Even a short walk outside can shift your mood.",
        ]
    }
}

# ── Load ML Model Once at Startup ─────────────────────────────
_model      = None
_vectorizer = None
_meta       = {}

def load_ml_assets():
    global _model, _vectorizer, _meta
    try:
        _model      = pickle.load(open(MODEL_PATH, 'rb'))
        _vectorizer = pickle.load(open(VEC_PATH,   'rb'))
        if os.path.exists(META_PATH):
            _meta = json.load(open(META_PATH))
        log.info(f"✅ Model loaded — version {_meta.get('version', '?')} | "
                 f"CV-F1={_meta.get('cv_f1_mean', '?')}")
    except FileNotFoundError:
        log.error("❌ Model files not found — run train_model.py first!")
    except Exception as e:
        log.error(f"❌ Failed to load model: {e}")

load_ml_assets()

# ── Text Preprocessing (identical to training) ─────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# ── SQLite Database ────────────────────────────────────────────
def get_db():
    """Return a singleton DB connection per Flask request context."""
    if 'db' not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row   # rows behave like dicts
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    """Create the history table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS mood_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            text_snippet TEXT    NOT NULL,
            prediction   TEXT    NOT NULL,
            badge        TEXT    NOT NULL,
            score        INTEGER NOT NULL,
            confidence   REAL    NOT NULL,
            journal      TEXT    DEFAULT ''
        )
    """)
    # Add journal column silently if upgrading from an older DB
    try:
        con.execute("ALTER TABLE mood_history ADD COLUMN journal TEXT DEFAULT ''")
        con.commit()
    except Exception:
        pass   # column already exists
    con.commit()
    con.close()
    log.info(f"Database ready at {DB_PATH}")

init_db()

# ── Helpers ────────────────────────────────────────────────────
def db_insert(timestamp, snippet, prediction, badge, score, confidence, journal=''):
    db = get_db()
    db.execute(
        "INSERT INTO mood_history "
        "(timestamp, text_snippet, prediction, badge, score, confidence, journal) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp, snippet, prediction, badge, score, confidence, journal)
    )
    db.commit()

def db_recent(limit=20):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM mood_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global _model, _vectorizer

    # Lazy reload on fail
    if _model is None or _vectorizer is None:
        load_ml_assets()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400

    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Text field is empty'}), 400

    # ── Predict ────────────────────────────────────────────────
    if _model and _vectorizer:
        processed       = preprocess(text)
        vec             = _vectorizer.transform([processed])
        label           = int(_model.predict(vec)[0])
        proba           = _model.predict_proba(vec)[0]
        confidence      = float(proba[label])

        info            = LABEL_MAP.get(label, LABEL_MAP[0])
        prediction      = info["prediction"]
        badge           = info["badge"]
        suggestions     = info["suggestions"]

        # Dynamic score: base ± confidence spread
        base            = info["base_score"]
        score           = int(base + (confidence - 0.60) * 30)
        score           = max(5, min(100, score))

        log.info(f"PREDICT | label={label} ({prediction}) | conf={confidence:.1%} | "
                 f"text='{text[:45]}...'")

        # ── Generate journal entry ──────────────────────────────
        journal = generate_journal(text, label, confidence, prediction)

    else:
        prediction  = "Model Not Loaded"
        badge       = "danger"
        score       = 0
        confidence  = 0.0
        suggestions = ["Run 'python train_model.py' to train the model."]
        journal     = "The model is not available right now. Please train it first."

    # ── Persist to DB ──────────────────────────────────────────
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snippet = text[:60] + "..." if len(text) > 60 else text

    # Fix 1: Skip insert if the last saved snippet is identical
    # (prevents duplicate rows from rapid button clicks)
    _db  = get_db()
    last = _db.execute(
        "SELECT text_snippet FROM mood_history ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if not last or last["text_snippet"] != snippet:
        db_insert(ts, snippet, prediction, badge, score, round(confidence, 4), journal)
        # Fix 3: Auto-trim — keep only the 50 most recent rows
        _db.execute(
            "DELETE FROM mood_history WHERE id NOT IN "
            "(SELECT id FROM mood_history ORDER BY id DESC LIMIT 50)"
        )
        _db.commit()
        log.info(f"DB saved + trimmed to <=50 rows")
    else:
        log.info(f"Duplicate text — skipping DB insert")


    return jsonify({
        'prediction':  prediction,
        'badge':       badge,
        'score':       score,
        'confidence':  round(confidence * 100, 1),
        'suggestions': suggestions,
        'timestamp':   ts,
        'journal':     journal
    })


@app.route('/api/history')
def api_history():
    """
    Returns recent mood history from SQLite.
    Used by Chart.js for real-time graph updates.
    Query param: ?limit=20
    """
    limit   = min(int(request.args.get('limit', 20)), 100)
    entries = db_recent(limit)
    return jsonify(entries)   # newest first


@app.route('/api/journal')
def api_journal():
    """
    Returns past journal entries (newest first).
    Query params: ?limit=10
    """
    limit = min(int(request.args.get('limit', 10)), 50)
    db    = get_db()
    rows  = db.execute(
        "SELECT id, timestamp, prediction, badge, score, journal "
        "FROM mood_history WHERE journal != '' "
        "ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])



@app.route('/api/stats')
def api_stats():
    """Dashboard summary statistics."""
    entries = db_recent(50)
    if not entries:
        return jsonify({'total': 0, 'happy': 0, 'stressed': 0, 'depressed': 0, 'avg_score': 0})

    happy     = sum(1 for e in entries if e['badge'] == 'success')
    stressed  = sum(1 for e in entries if e['badge'] == 'warning')
    depressed = sum(1 for e in entries if e['badge'] == 'danger')
    avg_score = round(sum(e['score'] for e in entries) / len(entries), 1)

    return jsonify({
        'total':     len(entries),
        'happy':     happy,
        'stressed':  stressed,
        'depressed': depressed,
        'avg_score': avg_score,
        'model_version': _meta.get('version', 'N/A'),
        'model_f1':      _meta.get('cv_f1_mean', 'N/A')
    })


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    db = get_db()
    db.execute("DELETE FROM mood_history")
    db.commit()
    log.info("History cleared by user.")
    return jsonify({'message': 'History cleared.'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
