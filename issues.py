"""Find the complaint themes behind negative reviews.

Each theme is a set of keyword patterns. A review counts once per theme it mentions,
so percentages can add up to more than 100%. This is transparent and fast, but it
only finds themes the rules describe.
"""
import re

ISSUE_RULES = [
    ("Delivery and shipping", r"deliver\w*|shipping|shipped|courier|\blate\b|delay\w*|took (?:too )?(?:long|days|weeks)|not (?:been )?delivered|didn'?t arrive"),
    ("Damaged or poor packaging", r"damag\w*|broken on arrival|arrived broken|packag\w*|packed|\btorn\b|crushed|dented|scratch\w*|dusty"),
    ("Poor build quality", r"cheap|flimsy|poor quality|low quality|\bquality\b|\bbuild\b|plastic|fragile|\bbroke\b|\bbroken\b|\bbreak\w*|stitch\w*|came apart|fell apart|\bloose\b|peel\w*|crack\w*|durab\w*|\btear\w*|\btore\b"),
    ("Defective or not working", r"defect\w*|faulty|not working|doesn'?t work|does not work|stopped working|stop working|\bdead\b|malfunction\w*|not functioning|won'?t turn on|doesn'?t turn on|\bdied\b|stopped (?:charging|working)"),
    ("Battery and charging", r"batter\w*|charg\w*|drain\w*"),
    ("Slow performance and crashes", r"slow\w*|\blag\w*|hang\w*|freez\w*|crash\w*|performance|\bspeed\b|sluggish|\bstuck\b"),
    ("Overheating", r"overheat\w*|\bheat\w*|\bhot\b|\bwarm\b|\bburn\w*"),
    ("Noise and sound quality", r"nois\w*|\bloud\w*|\bsound\b|\baudio\b|\bbass\b|buzz\w*|distort\w*|crackl\w*|microphone|\bmic\b"),
    ("Price and value", r"expensive|overpriced|costly|\bprice\w*|pricey|\bvalue\b|\bmoney\b|\bwaste\b|rip[- ]?off|not worth|\bworth\b"),
    ("Size, fit and appearance", r"\bsize\w*|\bsmall\w*|\blarge\b|\bbigger\b|\btight\b|\bfits?\b|colou?r|\blook\w*|\bdesign\b|\bshade\b"),
    ("Setup and instructions", r"set ?up|install\w*|instruction\w*|\bmanual\b|assembl\w*|assembly|confus\w*|hard to use|complicated|difficult to use"),
    ("Seller and customer service", r"\bseller\b|\bservice\b|\bsupport\b|refund\w*|\breturn\w*|replace\w*|customer care|\bcustomer\b|\breply\b|replied|respond\w*|helpline|warranty|\brude\b"),
    ("Wrong or missing item", r"\bwrong\b|\bdifferent\b|\bmissing\b|incomplete|not as (?:described|shown|per)|misleading|pictures?|\bphoto\b|\bimage\b|mismatch"),
    ("Smell and hygiene", r"smell\w*|odou?r|stink\w*|\bdirty\b|stain\w*|residue"),
    ("Skin irritation", r"\brash\w*|\bitch\w*|irritat\w*|allerg\w*|breakout\w*"),
]
COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in ISSUE_RULES]


def _clip(text, limit=200):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def find_issues(texts, sentiments, target="negative", max_examples=3, top=10):
    base = [t for t, s in zip(texts, sentiments) if s == target]
    total = len(base)
    rows = []
    for name, rx in COMPILED:
        hits = [t for t in base if rx.search(t)]
        if not hits:
            continue
        # Prefer readable examples: medium-length reviews first, then the rest.
        readable = [h for h in hits if 40 <= len(h) <= 220] or hits
        rows.append({
            "issue": name,
            "reviews": len(hits),
            "percent": round(len(hits) / total * 100, 1) if total else 0.0,
            "examples": [_clip(h) for h in readable[:max_examples]],
        })
    rows.sort(key=lambda r: r["reviews"], reverse=True)
    return {"base": target, "base_reviews": total, "issues": rows[:top]}
