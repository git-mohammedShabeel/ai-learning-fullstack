"""
OCR Engine — handles three input types:
  1. Digital PDFs  -> extract text directly (fast, accurate)
  2. Scanned PDFs  -> convert to image then OCR
  3. Images        -> EasyOCR for printed + handwritten text

Usage:
    from ocr_engine import OCREngine
    ocr = OCREngine()
    result = ocr.extract("path/to/file.pdf")
    print(result["text"])
"""

import os
import fitz          # PyMuPDF
import easyocr
from PIL import Image
import numpy as np


class OCREngine:
    def __init__(self, languages=["en"]):
        print("Loading EasyOCR model (first run downloads ~100MB)...")
        self.reader = easyocr.Reader(languages, gpu=True)
        print("OCR engine ready.")

    # ------------------------------------------------------------------
    # PUBLIC — main entry point
    # ------------------------------------------------------------------
    def extract(self, file_path: str) -> dict:
        """
        Auto-detects file type and extracts text.
        Returns:
            {
              "text":   str,        # full extracted text
              "pages":  int,        # number of pages (PDFs) or 1 (images)
              "method": str,        # "digital_pdf" | "scanned_pdf" | "image"
              "source": str         # original file path
            }
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            return self._extract_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PRIVATE — PDF handler
    # ------------------------------------------------------------------
    def _extract_pdf(self, pdf_path: str) -> dict:
        doc   = fitz.open(pdf_path)
        pages = len(doc)
        all_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if len(text) > 50:
                # Digital PDF — text layer exists, extract directly
                all_text.append(text)
                method = "digital_pdf"
            else:
                # Scanned PDF — render page as image then OCR
                pix  = page.get_pixmap(dpi=200)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                arr  = np.array(img)
                results = self.reader.readtext(arr, detail=0, paragraph=True)
                all_text.append("\n".join(results))
                method = "scanned_pdf"

            print(f"  Page {page_num+1}/{pages} done [{method}]")

        doc.close()
        return {
            "text":   "\n\n".join(all_text),
            "pages":  pages,
            "method": method,
            "source": pdf_path
        }

    # ------------------------------------------------------------------
    # PRIVATE — Image handler
    # ------------------------------------------------------------------
    def _extract_image(self, img_path: str) -> dict:
        img     = Image.open(img_path).convert("RGB")
        arr     = np.array(img)
        results = self.reader.readtext(arr, detail=0, paragraph=True)
        text    = "\n".join(results)
        return {
            "text":   text,
            "pages":  1,
            "method": "image",
            "source": img_path
        }


# ------------------------------------------------------------------
# Quick test — run this file directly to verify everything works
# python ocr_engine.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import glob, json

    base    = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-data")
    pdf_dir = os.path.join(base, "pdfs")
    hw_dir  = os.path.join(base, "handwritten")

    ocr = OCREngine()
    results_log = []

    # Test 1 — first 3 PDFs
    print("\n=== Testing PDFs ===")
    pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))[:3]
    for pdf in pdfs:
        r = ocr.extract(pdf)
        preview = r["text"][:120].replace("\n", " ")
        print(f"  [{r['method']}] {os.path.basename(pdf)}")
        print(f"  Pages: {r['pages']} | Preview: {preview}...")
        results_log.append({"file": os.path.basename(pdf), "method": r["method"],
                             "pages": r["pages"], "chars": len(r["text"])})

    # Test 2 — first 3 handwritten images
    print("\n=== Testing Handwritten Images ===")
    imgs = glob.glob(os.path.join(hw_dir, "*.png"))[:3]
    for img_path in imgs:
        r = ocr.extract(img_path)
        print(f"  [{r['method']}] {os.path.basename(img_path)}")
        print(f"  Extracted: {r['text'][:100]}")
        results_log.append({"file": os.path.basename(img_path), "method": r["method"],
                             "chars": len(r["text"])})

    # Save test log
    log_path = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-fullstack", "shared", "ocr_test_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nTest log saved -> {log_path}")
    print("\nOCR pipeline working. Ready for Day 4 (PDF text extraction + chunking).")