"""
save_json.py
Helper to format extracted product results as JSON and save them into the
output_json/ directory.
"""

import json
import os
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_result_json(upload_id: int, filename: str, attributes: dict,
                      raw_text: str, ocr_confidence: float, confidence_score: float) -> str:
    """
    Write the extraction result to a JSON file and return its path.
    """
    payload = {
        "upload_id": upload_id,
        "filename": filename,
        "extracted_at": datetime.utcnow().isoformat() + "Z",
        "attributes": attributes,
        "ocr_confidence": ocr_confidence,
        "confidence_score": confidence_score,
        "raw_text": raw_text,
    }

    safe_name = os.path.splitext(filename)[0]
    json_filename = f"{upload_id}_{safe_name}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return json_path
