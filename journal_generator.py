"""
journal_generator.py — Dynamic Reflective Journal Generator
────────────────────────────────────────────────────────────
Offline, no API needed.
Generates unique 5–8 sentence entries by:
  1. Randomly sampling sentence fragments from mood-specific pools
  2. Weaving in the user's own words and confidence level
  3. Appending a mood-matched closing with gentle encouragement

Each call produces a statistically unique entry.
"""

import random
import re
from datetime import datetime

# ── Shared helpers ────────────────────────────────────────────
def _pick(*items):
    """Pick one item at random from the arguments."""
    return random.choice(items)

def _clean(text: str) -> str:
    """Trim and ensure the text ends with a full stop."""
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def _extract_keywords(user_text: str, n: int = 3) -> list[str]:
    """Pull the most meaningful words from the user's input."""
    stopwords = {
        'i', 'me', 'my', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'did', 'the', 'a', 'an', 'and', 'or',
        'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 'that',
        'it', 'its', 'so', 'too', 'just', 'very', 'feel', 'feeling', 'really'
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', user_text.lower())
    keywords = [w for w in words if w not in stopwords]
    # De-duplicate while preserving order
    seen = set()
    unique = [w for w in keywords if not (w in seen or seen.add(w))]
    return unique[:n] if unique else ['your thoughts']

# ─────────────────────────────────────────────────────────────
# SENTENCE POOLS  (mood-specific)
# Each pool has multiple options so the generator rarely repeats
# ─────────────────────────────────────────────────────────────

_OPENERS = {
    0: [   # Happy / Normal
        "Today, your words carry the warmth of someone who is genuinely at peace.",
        "Reading what you have shared, there is a quiet, steady brightness in your perspective.",
        "It is clear from your reflection today that you are in a good place emotionally.",
        "Your entry today has an unmistakably positive energy flowing through it.",
        "There is something grounding and uplifting about the way you have expressed yourself today.",
    ],
    1: [   # Stressed / Anxious
        "Today's entry reveals the weight of a mind that is carrying quite a lot right now.",
        "Your words today hint at the kind of tension that builds when life asks too much at once.",
        "Reading your thoughts, it is evident that you are navigating a period of real pressure.",
        "There is a palpable sense of urgency in your writing today — you are clearly stretched thin.",
        "Your reflection today speaks honestly about the mental load you are currently managing.",
    ],
    2: [   # Depressed / Sad
        "Today's words carry a heaviness that deserves to be acknowledged with gentleness.",
        "Reading what you have shared, it is clear you are going through something deeply difficult.",
        "Your entry today reflects a moment of real pain, and that pain is completely valid.",
        "There is a quiet sadness woven through your words today that deserves compassionate attention.",
        "Your reflection today is honest and courageous — opening up about difficult feelings takes real strength.",
    ],
}

_REFLECTION = {
    0: [
        "Moments like these — when clarity outweighs confusion and ease outweighs struggle — are worth noticing and savouring.",
        "A positive mental state is not an accident; it often grows from small, consistent acts of self-care and connection.",
        "When you feel well, it is a wonderful time to observe what has been working and carry that forward.",
        "Contentment does not need a grand reason — sometimes it simply means that today's burdens feel manageable.",
        "Noticing when you feel okay, or even good, is itself a form of mindfulness worth practising daily.",
    ],
    1: [
        "Stress is most often a signal that something important to you is at risk — it is worth listening to, not suppressing.",
        "When anxiety rises, it rarely means things are unsolvable; more often, it means your mind needs a moment to pause.",
        "Pressure builds when demands outpace recovery — protecting even small pockets of rest matters enormously.",
        "The mind under stress often magnifies problems; grounding yourself in the present can gently restore perspective.",
        "It is easy under pressure to treat every task as equally urgent — but not everything deserves the same energy right now.",
    ],
    2: [
        "Sadness and low mood are not signs of weakness — they are your mind asking for care, the same way a body asks for rest.",
        "When everything feels heavy, it is not because you are incapable — it is because you are human and you are hurting.",
        "Difficult emotions do not last forever, even when they feel permanent in the moment you are experiencing them.",
        "You do not need to understand why you feel this way in order to show yourself compassion right now.",
        "Low periods are not the whole of who you are; they are a chapter, not the entire story.",
    ],
}

_KEYWORD_BRIDGE = {
    0: [
        "The way you mentioned {kw} suggests you have found some real meaning in that area of your life lately.",
        "It is encouraging to see {kw} appear in your writing — it reflects a grounded awareness of what matters to you.",
        "Your reference to {kw} speaks to the thoughtful way you are engaging with your daily experience.",
    ],
    1: [
        "The mention of {kw} in your entry points to one of the specific pressures weighing on you — acknowledging it is the first step.",
        "It sounds like {kw} has been taking up significant space in your mind; that is worth noticing without judgement.",
        "Your words around {kw} suggest this is one area where you may need to give yourself more patience and compassion.",
    ],
    2: [
        "The way you wrote about {kw} shows how deeply you are feeling things right now — that depth is part of who you are.",
        "Mentioning {kw} hints at something you are processing beneath the surface; give yourself the time to sit with it.",
        "Your sense of {kw} in this moment is real and valid — you deserve space to feel it without apology.",
    ],
}

_CONFIDENCE_NOTE = {
    'high': {
        0: "The strength of today's reading suggests your positive state is well-grounded, not fleeting.",
        1: "The reading today reflects a strong stress signal — this may be a good time to actively seek support.",
        2: "The depth of today's emotional reading suggests this is not simply a passing mood — please take it seriously.",
    },
    'medium': {
        0: "Your emotional state today sits in a healthy place, with some natural fluctuation — that is perfectly normal.",
        1: "Your stress level today is moderate — a timely reminder to check in with yourself before it intensifies.",
        2: "Today's reading suggests low mood, though there are signs of resilience in the way you have expressed yourself.",
    },
    'low': {
        0: "Even if today feels just 'okay' rather than great, that stability is its own quiet achievement.",
        1: "Your emotional state today shows some tension, though your own words suggest you are finding ways to cope.",
        2: "The reading today indicates some sadness, though the act of writing itself shows you have not given up.",
    },
}

_ENCOURAGEMENT = {
    0: [
        "Keep nurturing the habits and relationships that are clearly contributing to your wellbeing.",
        "Take a moment to appreciate where you are — you have earned this sense of steadiness.",
        "Continue to check in with yourself like this; self-awareness is the foundation of lasting mental health.",
        "Share this positive energy with someone around you — it has a way of multiplying.",
        "Use this good period to build the reserves that will support you through harder days ahead.",
    ],
    1: [
        "Be kind to yourself today — you do not need to solve everything at once.",
        "One small step: identify one thing you can set aside for now and allow yourself to breathe.",
        "Reaching out — to a friend, a mentor, or a professional — is a sign of strength, not weakness.",
        "Try to protect at least fifteen minutes today that belong entirely to rest, not productivity.",
        "Remember that asking for help is not admitting defeat — it is one of the most effective things you can do.",
    ],
    2: [
        "Please be gentle with yourself — you deserve the same compassion you would offer someone you love.",
        "If you feel comfortable doing so, consider speaking to someone you trust about how you are feeling.",
        "Small actions — a glass of water, five minutes outside, one honest conversation — can shift something.",
        "You do not have to feel better by tomorrow; just making it through today is enough for now.",
        "If you are in crisis, you do not have to face it alone — iCall: 9152987821 | Vandrevala: 1860-2662-345.",
    ],
}

_CLOSERS = {
    0: [
        "This journal entry is a small reminder that good days exist and that you are capable of experiencing them.",
        "Keep writing, keep reflecting — your consistency in checking in with yourself is a true act of self-care.",
        "Thank you for taking the time to pause and notice how you are doing today.",
    ],
    1: [
        "This entry is a checkpoint, not a verdict — you are allowed to feel this way and still move forward.",
        "Writing about it is already a form of processing; trust that something in you is working through this.",
        "You noticed how you were feeling and you did something about it — that matters more than you know.",
    ],
    2: [
        "Whatever today looks like, please know that what you are feeling will not always feel this heavy.",
        "This journal entry is evidence that you are still reaching out, still trying — and that means something.",
        "Your honesty with yourself is profound; please extend that same honesty when seeking support from others.",
    ],
}

# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def generate_journal(
    user_text: str,
    label: int,
    confidence: float,
    prediction: str
) -> str:
    """
    Generate a unique reflective journal entry.

    Args:
        user_text   : The raw text the user submitted.
        label       : 0 = Happy, 1 = Stressed, 2 = Depressed
        confidence  : Float 0–1 from model.predict_proba
        prediction  : Human-readable label string

    Returns:
        A multi-sentence journal string.
    """
    label = max(0, min(2, label))   # clamp to valid range

    # Confidence tier
    if confidence >= 0.65:
        tier = 'high'
    elif confidence >= 0.45:
        tier = 'medium'
    else:
        tier = 'low'

    # Extract user keywords to personalise the entry
    keywords = _extract_keywords(user_text)
    kw = random.choice(keywords) if keywords else 'what you shared'

    # ── Assemble sentences ────────────────────────────────────
    sentences = [
        # 1. Date + opener
        f"Journal entry — {datetime.now().strftime('%d %B %Y, %I:%M %p')}.",

        # 2. Mood-matched opening observation
        _pick(*_OPENERS[label]),

        # 3. Reflective insight about emotions
        _pick(*_REFLECTION[label]),

        # 4. Personalised keyword bridge
        random.choice(_KEYWORD_BRIDGE[label]).format(kw=kw),

        # 5. Confidence-aware note
        _CONFIDENCE_NOTE[tier][label],

        # 6. Random encouragement (sometimes 1, sometimes 2)
        _pick(*_ENCOURAGEMENT[label]),
    ]

    # 7. Occasionally add a second encouragement sentence for variety
    if random.random() > 0.4:
        second_enc = _pick(*_ENCOURAGEMENT[label])
        if second_enc != sentences[-1]:
            sentences.append(second_enc)

    # 8. Closing sentence
    sentences.append(_pick(*_CLOSERS[label]))

    # ── Join and clean ────────────────────────────────────────
    entry = ' '.join(_clean(s) for s in sentences)
    return entry
