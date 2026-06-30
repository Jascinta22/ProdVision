"""
parser.py
Extracts structured product attributes (brand, product name, weight, MRP,
manufacturing date, expiry date) from raw OCR text using a combination of
regular expressions and spaCy NLP.
"""

import re
import spacy

# --- spaCy model load (singleton) ------------------------------------------
SPACY_FULL_MODEL_LOADED = False
try:
    _nlp = spacy.load("en_core_web_sm")
    SPACY_FULL_MODEL_LOADED = True
except Exception:
    # Model not downloaded yet. Fall back to a blank English pipeline so the
    # app doesn't crash; brand/name extraction quality will be reduced until
    # `python -m spacy download en_core_web_sm` is run.
    _nlp = spacy.blank("en")

# --- Known brand dictionary (boosts recognition rate) ----------------------
KNOWN_BRANDS = [
    "Amul", "Britannia", "Nestle", "Nestlé", "Oreo", "Colgate", "Nivea",
    "Dove", "Pepsi", "Coca-Cola", "Coca Cola", "Parle", "Cadbury", "ITC",
    "Lays", "Lay's", "Maggi", "Patanjali", "Himalaya", "Dabur", "Lakme",
    "Lakmé", "Sunsilk", "Pantene", "Head & Shoulders", "Pond's", "Ponds",
    "Vaseline", "Johnson & Johnson", "Horlicks", "Bournvita", "Haldiram's",
    "Haldirams", "MDH", "Tata", "Surf Excel", "Ariel", "Rin", "Vim",
]

# --- Regex patterns ----------------------------------------------------------
# More lenient weight pattern: allows for spacing/corruption around unit
WEIGHT_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[/\-]?\s*(ml|g|kg|l|ltr|liters?|litres?|gm|gms)\b",
    re.IGNORECASE,
)

# Also try to match just digits followed by unit without strict word boundary
# to catch corrupted OCR like "100 9" -> "100 g"
WEIGHT_PATTERN_LOOSE = re.compile(
    r"(\d+(?:\.\d+)?)\s{0,3}(ml|g|kg|l|ltr|liters?|litres?|gm|gms)",
    re.IGNORECASE,
)

MRP_PATTERN = re.compile(
    r"(?:mrp|price|rs\.?|inr|₹)\s*[^a-zA-Z0-9\s]*\s*(\d+(?:[,.]\d{2})?)",
    re.IGNORECASE,
)

# Fallback MRP: just look for ₹ or Rs followed by a number
MRP_PATTERN_LOOSE = re.compile(
    r"[₹rs]{1,3}\s*(\d{1,4}(?:[,.]\d{2})?)",
    re.IGNORECASE,
)

# Matches DD/MM/YYYY, DD-MM-YYYY, MM/YYYY, "EXP MAR 2026", "MFG 01/2025" etc.
DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}[/\-]\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4})\b",
    re.IGNORECASE,
)
MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MFG_LABEL_PATTERN = re.compile(r"(?:mfg|mfd|manufactur(?:ed|ing))\D{0,15}", re.IGNORECASE)
EXP_LABEL_PATTERN = re.compile(r"(?:exp|expiry|best before|use by)\D{0,15}", re.IGNORECASE)
BEST_BEFORE_PATTERN = re.compile(r"(?:best before|expiry|exp|use by)\D{0,20}(\d+)\s*(months?|yrs?|years?)", re.IGNORECASE)
PRODUCT_KEYWORDS = [
    "butter", "milk", "ghee", "cheese", "cream", "yogurt", "yoghurt",
    "chocolate", "biscuit", "cookies", "oil", "soap", "shampoo",
    "toothpaste", "detergent", "jam", "sauce", "pickle", "tea", "coffee",
    "chips", "snacks", "powder", "masala", "noodles", "pasta"
]
NON_PRODUCT_TERMS = [
    "mrp", "price", "rs", "inr", "₹", "exp", "mfg", "mfd", "pkd",
    "weight", "net", "qty", "packed", "batch", "lic", "no:", "vol",
    "ml", "g", "kg", "pcs", "pasteur", "pasteurised", "processed",
    "manufactured", "manufacture", "made in", "made by", "seller",
    "distributor", "company", "brand"
]


def _extract_weight(text: str):
    match = WEIGHT_PATTERN.search(text)
    if not match:
        # Fallback: try looser pattern to catch corrupted OCR
        match = WEIGHT_PATTERN_LOOSE.search(text)
    if match:
        return f"{match.group(1)} {match.group(2).lower()}"
    return None


def _extract_mrp(text: str):
    match = MRP_PATTERN.search(text)
    if not match:
        # Fallback: try looser pattern with just ₹ or Rs
        match = MRP_PATTERN_LOOSE.search(text)
    if match:
        return match.group(1).replace(",", "")
    return None


def _extract_labeled_date(text: str, label_pattern: re.Pattern):
    """Find a date that appears shortly after a label like MFG/EXP."""
    label_match = label_pattern.search(text)
    if not label_match:
        return None
    window = text[label_match.end(): label_match.end() + 20]
    date_match = DATE_PATTERN.search(window)
    return date_match.group(0) if date_match else None


def _infer_expiry_from_best_before(text: str, mfg_date: str):
    """Infer an expiry date from phrases like 'best before 12 months'."""
    match = BEST_BEFORE_PATTERN.search(text)
    if not match:
        return None

    months_to_add = int(match.group(1))
    if not mfg_date:
        return f"{months_to_add} months from manufacture"

    if re.fullmatch(r"\d{1,2}[/\-]\d{4}", mfg_date):
        day, month_year = mfg_date.split("/")
        day = int(day)
        month = int(month_year[:2]) if len(month_year) > 2 else None
        year = int(month_year[3:]) if len(month_year) > 2 else None
        if month is None or year is None:
            return None
        month += months_to_add
        year += (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return f"{day:02d}/{month:02d}/{year}"

    if re.fullmatch(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", mfg_date):
        parts = re.split(r"[/\-]", mfg_date)
        if len(parts) == 3:
            day, month, year = map(int, parts)
            month += months_to_add
            year += (month - 1) // 12
            month = ((month - 1) % 12) + 1
            return f"{day:02d}/{month:02d}/{year}"

    return None


def _extract_dates(text: str):
    mfg_date = _extract_labeled_date(text, MFG_LABEL_PATTERN)
    exp_date = _extract_labeled_date(text, EXP_LABEL_PATTERN)

    if not exp_date:
        exp_date = _infer_expiry_from_best_before(text, mfg_date)

    # Fallback: if labels weren't found near a date, grab the first two
    # distinct dates found anywhere in the text as best-effort guesses.
    if not mfg_date or not exp_date:
        all_dates = DATE_PATTERN.findall(text)
        if not mfg_date and len(all_dates) >= 1:
            mfg_date = all_dates[0]
        if not exp_date and len(all_dates) >= 2:
            exp_date = all_dates[1]

    return mfg_date, exp_date


def _extract_brand(text: str, doc):
    # 1. Try direct match against the known brand dictionary (case-insensitive).
    lowered = text.lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in lowered:
            return brand

    # 2. Fall back to spaCy NER for ORG entities.
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text.strip()

    return None


def _extract_product_name(text: str, doc, brand: str):
    lowered_text = text.lower()

    keyword_match = None
    for keyword in PRODUCT_KEYWORDS:
        if keyword in lowered_text:
            keyword_match = keyword
            break
    if keyword_match:
        return keyword_match.capitalize()

    # Use the longest noun chunk only if a full spacy model is loaded.
    if SPACY_FULL_MODEL_LOADED:
        candidates = [chunk.text.strip() for chunk in getattr(doc, "noun_chunks", [])]
        candidates = [c for c in candidates if c and (not brand or brand.lower() not in c.lower())]
        candidates = [c for c in candidates if not any(kw in c.lower() for kw in ["mrp", "mfg", "exp", "rs.", "price", "net vol", "weight"])]
        if candidates:
            return max(candidates, key=len)

    # Fallback: clean lines and parse layout
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    clean_lines = []
    brand_line_idx = -1
    brand_lower = brand.lower() if brand else ""

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(nk in line_lower for nk in NON_PRODUCT_TERMS):
            continue
        if re.match(r"^[\d/\.\-\s:]+$", line):
            continue
        if brand_lower and brand_lower in line_lower:
            line_lower = line_lower.replace(brand_lower, "")
        if not re.search(r"[a-z]", line_lower):
            continue

        clean_lines.append((idx, line))
        if brand_lower and brand_lower in line_lower:
            brand_line_idx = idx

    # Prefer the first meaningful line after the brand/manufacturer line.
    if brand_line_idx != -1:
        for idx, line in clean_lines:
            if idx > brand_line_idx:
                if len(line.split()) <= 6:
                    return line

    for idx, line in clean_lines:
        if brand_lower and brand_lower == line.lower():
            continue
        if len(line.split()) <= 6:
            return line

    if lines:
        return lines[0]

    return "Product"


def parse_attributes(text: str) -> dict:
    """
    Parse raw OCR text and return a dict of extracted attributes:
    brand, product_name, weight, mrp, mfg_date, exp_date.
    """
    doc = _nlp(text)

    brand = _extract_brand(text, doc)
    product_name = _extract_product_name(text, doc, brand)
    weight = _extract_weight(text)
    mrp = _extract_mrp(text)
    mfg_date, exp_date = _extract_dates(text)

    return {
        "brand": brand,
        "product_name": product_name,
        "weight": weight,
        "mrp": mrp,
        "mfg_date": mfg_date,
        "exp_date": exp_date,
    }


def compute_confidence_score(ocr_confidence: float, attributes: dict) -> float:
    """
    Combine Pytesseract's OCR confidence with a heuristic based on how many
    of the key attributes were successfully extracted.

    Final score = 60% OCR confidence + 40% extraction completeness (as a
    percentage of the 6 tracked fields that were found).
    """
    total_fields = len(attributes)
    found_fields = sum(1 for v in attributes.values() if v)
    completeness = (found_fields / total_fields) * 100 if total_fields else 0

    score = (0.6 * ocr_confidence) + (0.4 * completeness)
    return round(score, 2)
