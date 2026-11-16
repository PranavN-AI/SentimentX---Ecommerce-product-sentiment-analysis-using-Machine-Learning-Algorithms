"""Generate a large labelled review corpus for SentimentX.

Reviews are assembled from phrase pools, with negation ("not good"), intensifiers,
mixed sentences and typos, so the model sees varied wording. To train on real reviews
instead, put a CSV with `review,sentiment` columns in data/ and run `python train.py --data <file>`.
"""
import csv
import random
from pathlib import Path

SEED = 7
PER_CLASS = 15000
LABEL_NOISE = 0.06  # share of labels flipped, like disagreeing human annotators
OUT = Path(__file__).parent / "data" / "reviews_large.csv"

PRODUCTS = [
    "headphones", "phone", "laptop", "blender", "running shoes", "backpack", "water bottle",
    "smartwatch", "kettle", "office chair", "jacket", "speaker", "camera", "novel",
    "vacuum cleaner", "keyboard", "desk lamp", "mattress", "toaster", "earbuds", "tablet",
    "yoga mat", "monitor", "rice cooker",
]
ASPECTS = [
    "quality", "battery", "delivery", "packaging", "sound", "fit", "design", "price", "build",
    "setup", "customer service", "performance", "size", "colour", "manual", "comfort",
    "durability", "display", "charging", "stitching",
]
POS = ["excellent", "fantastic", "solid", "impressive", "great", "superb", "lovely", "reliable",
       "amazing", "wonderful", "top notch", "well made", "smooth", "delightful", "brilliant", "good"]
NEG = ["terrible", "poor", "disappointing", "flimsy", "awful", "cheap", "unreliable", "overpriced",
       "broken", "frustrating", "useless", "shoddy", "annoying", "defective", "bad", "weak"]
NEU = ["average", "okay", "decent", "fine", "acceptable", "alright", "mediocre", "so so",
       "middling", "unremarkable", "passable", "ordinary"]
INTENS = ["really", "very", "extremely", "truly", "quite", "so", "", "", ""]
OPEN_POS = ["Honestly, ", "After a month of use, ", "Ordered this last week and ",
            "My friend recommended it, and ", "", "", ""]
OPEN_NEG = ["Honestly, ", "After a week, ", "Ordered this and ", "I expected more, ", "", "", ""]
OPEN_NEU = ["Honestly, ", "After a few weeks, ", "Overall, ", "", "", ""]

POS_TPL = [
    "The {asp} is {i}{pos}.",
    "I love the {asp} of this {prod}.",
    "{open}the {prod} is {i}{pos}, would recommend.",
    "Works great and the {asp} is {pos}.",
    "Not {neg} at all, the {asp} is {i}{pos}.",
    "Would buy this {prod} again, {asp} is {pos}.",
    "Pleasantly surprised, the {asp} is {pos}.",
]
NEG_TPL = [
    "The {asp} is {i}{neg}.",
    "I am disappointed with the {asp} of this {prod}.",
    "{open}the {prod} is {i}{neg}. Returning it.",
    "Do not buy this {prod}, the {asp} is {neg}.",
    "Not {pos} at all, the {asp} is {neg}.",
    "Broke within a week. {asp_c} is {neg}.",
    "Waste of money. The {asp} is {neg}.",
    "Would not recommend, {asp} is {neg}.",
]
NEU_TPL = [
    "The {asp} is {neu}, nothing more.",
    "The {prod} is {neu}. Some good points, some bad.",
    "{open}the {prod} is {neu}, does the job.",
    "Mixed feelings about the {asp}.",
    "Not {neg} but not {pos} either, the {asp} is {neu}.",
    "{asp_c} is {neu}. Expected a bit more.",
    "Okay {prod}, the {asp} is {neu}.",
]


def fill(tpl, opens, rng):
    asp = rng.choice(ASPECTS)
    intens = rng.choice(INTENS)
    return tpl.format(
        asp=asp,
        asp_c=asp[0].upper() + asp[1:],
        prod=rng.choice(PRODUCTS),
        i=(intens + " ") if intens else "",
        pos=rng.choice(POS),
        neg=rng.choice(NEG),
        neu=rng.choice(NEU),
        open=rng.choice(opens),
    )


def noise(text, rng):
    if rng.random() < 0.08:
        text = text.lower()
    if rng.random() < 0.05:
        text += "!!"
    if rng.random() < 0.06:
        words = text.split(" ")
        k = rng.randrange(len(words))
        w = words[k]
        if len(w) > 3:
            j = rng.randrange(len(w) - 1)
            words[k] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        text = " ".join(words)
    return text


def main():
    rng = random.Random(SEED)
    spec = {
        "positive": (POS_TPL, OPEN_POS, lambda: rng.choice(["4", "5"])),
        "neutral": (NEU_TPL, OPEN_NEU, lambda: "3"),
        "negative": (NEG_TPL, OPEN_NEG, lambda: rng.choice(["1", "2"])),
    }
    rows, seen = [], set()
    for label, (tpls, opens, stars) in spec.items():
        made = 0
        while made < PER_CLASS:
            text = noise(fill(rng.choice(tpls), opens, rng), rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"review": text, "sentiment": label, "stars": stars()})
            made += 1
    for r in rows:
        if rng.random() < LABEL_NOISE:
            r["sentiment"] = rng.choice([l for l in spec if l != r["sentiment"]])
    rng.shuffle(rows)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["review", "sentiment", "stars"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} reviews to {OUT}")


if __name__ == "__main__":
    main()
