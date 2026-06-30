"""
app.py
Main FastAPI application. Coordinates upload, preprocessing, OCR, parsing,
database storage, JSON output, and serves the frontend dashboard.
"""

import os
import shutil
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

import database
import models
from preprocess import preprocess_image
from ocr import run_ocr
from parser import parse_attributes, compute_confidence_score
from save_json import save_result_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app = FastAPI(title="ProdVision", description="Product Information Extraction API")

# Serve uploaded images and static assets
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _merge_extraction_results(processed_results: list[dict]) -> dict:
    """Merge several per-image results into one combined result for a single product."""
    if not processed_results:
        return {}

    fields = ["brand", "product_name", "weight", "mrp", "mfg_date", "exp_date"]
    merged_attributes = {}

    for field in fields:
        for result in sorted(
            processed_results,
            key=lambda item: (item.get("confidence_score", 0), item.get("ocr_confidence", 0)),
            reverse=True,
        ):
            value = result.get("attributes", {}).get(field)
            if value:
                merged_attributes[field] = value
                break
        if field not in merged_attributes:
            merged_attributes[field] = None

    avg_confidence = round(
        sum(item.get("confidence_score", 0) for item in processed_results) / len(processed_results),
        2,
    )
    avg_ocr_confidence = round(
        sum(item.get("ocr_confidence", 0) for item in processed_results) / len(processed_results),
        2,
    )

    return {
        "filename": ", ".join(item["filename"] for item in processed_results),
        "image_urls": [item["image_url"] for item in processed_results],
        "attributes": merged_attributes,
        "ocr_confidence": avg_ocr_confidence,
        "confidence_score": avg_confidence,
        "raw_text": "\n\n--- NEXT IMAGE ---\n\n".join(item["raw_text"] for item in processed_results),
    }


@app.on_event("startup")
def on_startup():
    database.init_db()


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the HTML frontend dashboard."""
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload")
async def upload_image(files: List[UploadFile] = File(...)):
    """
    Upload one or more product images, run preprocessing -> OCR -> parsing
    for each, then merge them into one combined result for the same product.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    batch_id = uuid.uuid4().hex[:12]
    processed_results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        unique_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            processed_path = os.path.join(PROCESSED_DIR, unique_name)
            processed_image = preprocess_image(saved_path, save_path=processed_path)

            ocr_result = run_ocr(processed_image)
            LOW_CONFIDENCE_THRESHOLD = 40.0
            if ocr_result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                fallback_image = preprocess_image(
                    saved_path, save_path=processed_path, return_grayscale_fallback=True
                )
                fallback_result = run_ocr(fallback_image)
                if fallback_result["confidence"] > ocr_result["confidence"]:
                    ocr_result = fallback_result

            raw_text = ocr_result["text"]
            ocr_confidence = ocr_result["confidence"]
            attributes = parse_attributes(raw_text)
            confidence_score = compute_confidence_score(ocr_confidence, attributes)

            processed_results.append({
                "filename": file.filename,
                "image_url": f"/uploads/{unique_name}",
                "attributes": attributes,
                "ocr_confidence": ocr_confidence,
                "confidence_score": confidence_score,
                "raw_text": raw_text,
            })
        except Exception:
            continue

    if not processed_results:
        raise HTTPException(status_code=400, detail="No valid images could be processed")

    merged_result = _merge_extraction_results(processed_results)

    upload_id = models.insert_upload(
        batch_id=batch_id,
        filename=merged_result["filename"],
        filepath=processed_results[0]["image_url"],
        image_side=None,
        raw_text=merged_result["raw_text"],
        ocr_confidence=merged_result["ocr_confidence"],
    )

    json_path = save_result_json(
        upload_id=upload_id,
        filename=merged_result["filename"],
        attributes=merged_result["attributes"],
        raw_text=merged_result["raw_text"],
        ocr_confidence=merged_result["ocr_confidence"],
        confidence_score=merged_result["confidence_score"],
    )

    product_id = models.insert_product(
        upload_id=upload_id,
        attributes=merged_result["attributes"],
        confidence_score=merged_result["confidence_score"],
        json_path=json_path,
    )

    merged_result.update({
        "id": product_id,
        "upload_id": upload_id,
        "image_url": processed_results[0]["image_url"],
    })

    return JSONResponse({
        "batch_id": batch_id,
        "file_count": len(files),
        "result": merged_result,
        "individual_results": processed_results,
    })


@app.get("/products")
def list_products():
    """Return a list of all previously processed product extractions."""
    return models.get_all_products()


@app.get("/products/{product_id}/json")
def download_json(product_id: int):
    """Download the saved JSON metadata for a given product."""
    product = models.get_product_by_id(product_id)
    if not product or not product.get("json_path") or not os.path.exists(product["json_path"]):
        raise HTTPException(status_code=404, detail="JSON file not found for this product")
    return FileResponse(
        product["json_path"],
        media_type="application/json",
        filename=os.path.basename(product["json_path"]),
    )


@app.get("/products/{product_id}/pdf")
def download_pdf(product_id: int):
    """Generate and download a PDF summary of the product metadata."""
    product = models.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    pdf_dir = os.path.join(BASE_DIR, "output_pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"product_{product_id}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("ProdVision — Product Extraction Report", styles["Title"]))
    elements.append(Spacer(1, 10 * mm))
    elements.append(Paragraph(f"Source file: {product.get('filename', '-')}", styles["Normal"]))
    elements.append(Paragraph(f"Extracted at: {product.get('created_at', '-')}", styles["Normal"]))
    elements.append(Spacer(1, 8 * mm))

    table_data = [
        ["Field", "Value"],
        ["Brand", product.get("brand") or "-"],
        ["Product Name", product.get("product_name") or "-"],
        ["Weight / Volume", product.get("weight") or "-"],
        ["MRP", product.get("mrp") or "-"],
        ["Manufacturing Date", product.get("mfg_date") or "-"],
        ["Expiry Date", product.get("exp_date") or "-"],
        ["Confidence Score", f"{product.get('confidence_score', '-')} %"],
    ]

    table = Table(table_data, colWidths=[60 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)

    doc.build(elements)

    return FileResponse(pdf_path, media_type="application/pdf",
                         filename=f"product_{product_id}.pdf")


@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    """Delete a single product and its associated upload."""
    success = models.delete_product(product_id)
    if not success:
        raise HTTPException(status_code=404, detail="Product not found")
    return JSONResponse({"message": f"Product {product_id} deleted successfully"})


@app.delete("/products")
def delete_all_products():
    """Delete all products and uploads (clear entire history)."""
    count = models.delete_all_products()
    return JSONResponse({
        "message": f"Successfully deleted {count} products and all associated uploads",
        "deleted_count": count
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
