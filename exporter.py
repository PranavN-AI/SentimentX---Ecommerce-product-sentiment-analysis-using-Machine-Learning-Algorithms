"""Build an Excel workbook from a SentimentX analysis."""
from datetime import datetime
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

LABELS = ["positive", "neutral", "negative"]
SENT_FILL = {"positive": "D9F2E6", "neutral": "FDF0CC", "negative": "FBDDE2"}
HEAD_FILL = PatternFill("solid", fgColor="1B2030")
HEAD_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=16)
WRAP = Alignment(wrap_text=True, vertical="top")


def _cell(ws, row, col, value, font=None, fill=None, wrap=False, fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    if isinstance(value, str):
        cell.data_type = "s"          # keep text such as "=..." as plain text, never a formula
    if font:
        cell.font = font
    if fill:
        cell.fill = fill
    if wrap:
        cell.alignment = WRAP
    if fmt:
        cell.number_format = fmt
    return cell


def _header(ws, row, names):
    for i, name in enumerate(names, start=1):
        _cell(ws, row, i, name, font=HEAD_FONT, fill=HEAD_FILL)


def _widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _pct(v):
    return "n/a" if v is None else f"{v:.2f}%"


def build_workbook(payload):
    analysis = payload["analysis"]
    file_name = str(payload.get("file_name") or "reviews.csv")
    metrics = analysis.get("metrics") or {}
    reviews = analysis.get("reviews") or []
    issues = analysis.get("issues") or {}
    total = analysis.get("total", len(reviews))
    wb = Workbook()

    # ---- Summary ----
    ws = wb.active
    ws.title = "Summary"
    _cell(ws, 1, 1, "SentimentX analysis report", font=TITLE_FONT)
    info = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Source file", file_name),
        ("Total reviews", total),
        ("Model", "TF-IDF (words + characters) with calibrated Linear SVM"),
        ("Training data", metrics.get("dataset", "")),
        ("Accuracy on held-out test set", _pct(metrics.get("accuracy", 0) * 100 if metrics.get("accuracy") is not None else None)),
        ("Macro F1 on held-out test set", metrics.get("macro_f1")),
        ("Agreement with star ratings", _pct(analysis.get("star_agreement"))),
        ("Accuracy on your labels", _pct(analysis.get("user_accuracy"))),
        ("Average sentiment score (-1 to +1)", round(analysis.get("mean_score", 0), 3)),
    ]
    for r, (k, v) in enumerate(info, start=3):
        _cell(ws, r, 1, k, font=Font(bold=True))
        _cell(ws, r, 2, v)
    r = 3 + len(info) + 1
    _header(ws, r, ["Sentiment", "Reviews", "Share of reviews"])
    for i, row in enumerate(analysis.get("distribution", []), start=r + 1):
        lab = row["label"]
        _cell(ws, i, 1, lab.capitalize(), fill=PatternFill("solid", fgColor=SENT_FILL[lab]))
        _cell(ws, i, 2, row["count"])
        _cell(ws, i, 3, row["percent"] / 100, fmt="0.0%")
    _widths(ws, [36, 60, 18])

    # ---- Reviews ----
    ws = wb.create_sheet("Reviews")
    cols = ["Review", "Sentiment", "Confidence", "Score", "Stars", "Matches stars"]
    _header(ws, 1, cols)
    for i, rv in enumerate(reviews, start=2):
        expected = rv.get("expected")
        match = "" if expected is None else ("yes" if expected == rv["sentiment"] else "no")
        _cell(ws, i, 1, rv["text"], wrap=True)
        _cell(ws, i, 2, rv["sentiment"], fill=PatternFill("solid", fgColor=SENT_FILL[rv["sentiment"]]))
        _cell(ws, i, 3, rv["confidence"], fmt="0.0%")
        _cell(ws, i, 4, rv["score"], fmt="0.000")
        _cell(ws, i, 5, rv["stars"] if rv.get("stars") is not None else "")
        _cell(ws, i, 6, match)
    last = max(len(reviews) + 1, 2)
    if reviews:
        table = Table(displayName="Reviews", ref=f"A1:F{last}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=True)
        ws.add_table(table)
    ws.freeze_panes = "A2"
    _widths(ws, [90, 14, 14, 10, 8, 15])

    # ---- Top issues ----
    ws = wb.create_sheet("Top issues")
    _cell(ws, 1, 1, f"Complaint themes in {issues.get('base_reviews', 0)} negative reviews. "
                    "A review can mention more than one theme.", font=Font(italic=True))
    _header(ws, 3, ["Rank", "Issue", "Negative reviews mentioning it", "% of negative reviews", "Example review"])
    for i, row in enumerate(issues.get("issues", []), start=4):
        example = row["examples"][0] if row.get("examples") else ""
        _cell(ws, i, 1, i - 3)
        _cell(ws, i, 2, row["issue"], font=Font(bold=True))
        _cell(ws, i, 3, row["reviews"])
        _cell(ws, i, 4, row["percent"] / 100, fmt="0.0%")
        _cell(ws, i, 5, example, wrap=True)
    _widths(ws, [8, 32, 28, 22, 90])

    # ---- Top words ----
    ws = wb.create_sheet("Top words")
    _header(ws, 1, ["Sentiment", "Rank", "Word"])
    row_no = 2
    for lab in LABELS:
        for rank, word in enumerate(analysis.get("top_words", {}).get(lab, []), start=1):
            _cell(ws, row_no, 1, lab.capitalize(), fill=PatternFill("solid", fgColor=SENT_FILL[lab]))
            _cell(ws, row_no, 2, rank)
            _cell(ws, row_no, 3, word)
            row_no += 1
    _widths(ws, [16, 10, 24])

    # ---- Model quality ----
    ws = wb.create_sheet("Model quality")
    _header(ws, 1, ["Class", "Precision", "Recall", "F1", "Test reviews"])
    per_class = metrics.get("per_class", {})
    for i, lab in enumerate(LABELS, start=2):
        p = per_class.get(lab, {})
        _cell(ws, i, 1, lab.capitalize(), fill=PatternFill("solid", fgColor=SENT_FILL[lab]))
        _cell(ws, i, 2, p.get("precision"), fmt="0.000")
        _cell(ws, i, 3, p.get("recall"), fmt="0.000")
        _cell(ws, i, 4, p.get("f1"), fmt="0.000")
        _cell(ws, i, 5, p.get("support"))
    r = 7
    _cell(ws, r, 1, "Confusion matrix (rows: true label, columns: predicted)", font=Font(bold=True))
    _header(ws, r + 1, ["True \\ Predicted"] + [l.capitalize() for l in LABELS])
    for i, lab in enumerate(LABELS):
        _cell(ws, r + 2 + i, 1, lab.capitalize(), font=Font(bold=True))
        for j, count in enumerate((metrics.get("confusion") or [[0, 0, 0]] * 3)[i]):
            _cell(ws, r + 2 + i, 2 + j, count)
    _widths(ws, [22, 14, 14, 14, 16])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
