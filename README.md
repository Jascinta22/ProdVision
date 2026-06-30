# ProdVision — Product Information Extraction

Extracts structured metadata (brand, product name, weight, MRP, manufacturing
date, expiry date) from photos of product packaging using OpenCV preprocessing,
Tesseract OCR, and spaCy/Regex parsing. Results are stored in SQLite + JSON and
shown on a glassmorphic dashboard with PDF/JSON export.

## Project layout

```
prodvision/
├── app.py                 # FastAPI app & endpoints
├── database.py            # SQLite connection + schema
├── models.py               # DB read/write operations
├── preprocess.py           # OpenCV image cleanup pipeline
├── ocr.py                  # Pytesseract wrapper
├── parser.py                # Regex + spaCy attribute extraction
├── save_json.py             # JSON output writer
├── requirements.txt
├── templates/
│   └── index.html          # Dashboard UI
├── static/
│   └── styles.css           # Dashboard styling
├── utils/
│   └── test_pipeline.py     # Unit tests
├── uploads/                  # Saved original + preprocessed images (created at runtime)
├── output_json/               # Saved JSON extraction results (created at runtime)
├── output_pdf/                 # Generated PDF reports (created at runtime)
└── db/
    └── prodvision.db           # SQLite database (created at runtime)
```

## 1. Install Tesseract OCR

**Windows:**
```powershell
winget install UB-Mannheim.TesseractOCR --silent --accept-package-agreements --accept-source-agreements
```
If this fails, download the installer manually from the
[UB-Mannheim Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki) and
add it to your PATH. `ocr.py` also falls back to the default install path
`C:\Program Files\Tesseract-OCR\tesseract.exe` automatically. You can override
the path explicitly with an environment variable:

```powershell
set TESSERACT_CMD=C:\path\to\tesseract.exe
```

**macOS:** `brew install tesseract`
**Linux:** `sudo apt install tesseract-ocr`

## 2. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 3. Run the server

```bash
python -m uvicorn app:app --reload
```

Then open **http://localhost:8000** in your browser.

## 4. Run tests

```bash
python -m pytest utils/test_pipeline.py -v
```

## API endpoints

| Method | Path                      | Description                                   |
|--------|---------------------------|------------------------------------------------|
| GET    | `/`                        | Dashboard UI                                   |
| POST   | `/upload`                   | Upload an image, run full extraction pipeline |
| GET    | `/products`                  | List all previous extractions                 |
| GET    | `/products/{id}/pdf`          | Download a PDF summary                       |
| GET    | `/products/{id}/json`          | Download the raw JSON result                |

## Notes

- The brand dictionary in `parser.py` (`KNOWN_BRANDS`) can be extended with
  any additional brand names you want spaCy/regex matching to prioritize.
- Confidence score = `0.6 × OCR word confidence + 0.4 × % of fields successfully extracted`.
- If `en_core_web_sm` hasn't been downloaded yet, the app still runs using a
  blank spaCy pipeline (brand/name detection relies more heavily on the
  dictionary and regex fallback in that case).
"# ProdVision" 
