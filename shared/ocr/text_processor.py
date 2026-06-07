"""
Text Processor — cleans and chunks raw text from OCR/PDF extraction.

Pipeline:
    raw text -> clean -> split into chunks -> save as JSONL

Usage:
    from text_processor import TextProcessor
    tp = TextProcessor()
    chunks = tp.process("raw messy text here...")
    # or process a file directly:
    chunks = tp.process_file("path/to/file.pdf")

Each chunk is a dict:
    {
      "id":       "doc_001_chunk_003",
      "text":     "cleaned chunk text...",
      "tokens":   142,
      "source":   "paper_01.pdf",
      "page":     2
    }
"""

import os
import re
import json
import glob
import unicodedata
from pathlib import Path


class TextProcessor:

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def process(self, raw_text: str,
                source: str = "unknown",
                page: int = -1) -> list:
        cleaned = self._clean(raw_text)
        if not cleaned:
            return []
        chunks = self._chunk(cleaned)
        return self._label(chunks, source, page)

    def process_file(self, file_path: str) -> list:
        try:
            from ocr_engine import OCREngine
        except ImportError:
            raise ImportError("ocr_engine.py must be in the same folder.")
        ocr    = OCREngine()
        result = ocr.extract(file_path)
        return self.process(result["text"],
                            source=os.path.basename(file_path),
                            page=result["pages"])

    def save_chunks(self, chunks: list, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"  Saved {len(chunks)} chunks -> {out_path}")

    def load_chunks(self, jsonl_path: str) -> list:
        chunks = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    # ------------------------------------------------------------------
    # PRIVATE — cleaning (regex-safe version)
    # ------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        # Normalise unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove null bytes and control chars (except \n \t)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Cc"
            or ch in ("\n", "\t")
        )

        # Fix common OCR ligature artefacts (plain string replacements,
        # no regex so special chars can never break this)
        replacements = {
            "\ufb01": "fi",   # ﬁ
            "\ufb02": "fl",   # ﬂ
            "\u2018": "'",    # left single quote
            "\u2019": "'",    # right single quote
            "\u201c": '"',    # left double quote
            "\u201d": '"',    # right double quote
            "\u2013": "-",    # en dash
            "\u2014": "--",   # em dash
            "\u00ad": "",     # soft hyphen
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove lone page numbers (a line containing only digits)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are purely digits (page numbers)
            if stripped.isdigit():
                continue
            # Skip lines that are purely separator chars
            if stripped and all(c in "-=_|+ \t" for c in stripped):
                continue
            # Skip very short lines that are likely headers/footers
            if len(stripped) < 3:
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Collapse multiple spaces and excessive blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ------------------------------------------------------------------
    # PRIVATE — chunking (sliding window over sentences)
    # ------------------------------------------------------------------
    def _chunk(self, text: str) -> list:
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks  = []
        current = []
        cur_len = 0

        for sentence in sentences:
            words = sentence.split()
            wlen  = len(words)

            if cur_len + wlen > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap = " ".join(current).split()[-self.chunk_overlap:]
                current = list(overlap)
                cur_len = len(overlap)

            current.append(sentence)
            cur_len += wlen

        if current:
            chunks.append(" ".join(current))

        # Filter noise chunks (too short to be useful)
        chunks = [c for c in chunks if len(c.split()) >= 20]
        return chunks

    # ------------------------------------------------------------------
    # PRIVATE — labelling
    # ------------------------------------------------------------------
    def _label(self, chunks: list, source: str, page: int) -> list:
        stem = Path(source).stem[:20]
        # Sanitise stem so it only contains safe characters
        stem = re.sub(r"[^\w\-]", "_", stem)
        return [
            {
                "id":     f"{stem}_chunk_{i:04d}",
                "text":   chunk,
                "tokens": len(chunk.split()),
                "source": source,
                "page":   page,
            }
            for i, chunk in enumerate(chunks)
        ]


# ------------------------------------------------------------------
# Self-test — python text_processor.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    base    = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-data")
    pdf_dir = os.path.join(base, "pdfs")


    # Full-stack:
    out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-fullstack", "shared", "processed")
    
    os.makedirs(out_dir, exist_ok=True)
    tp   = TextProcessor(chunk_size=400, chunk_overlap=50)
    pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))[:10]

    if not pdfs:
        print("No PDFs found. Check pdf_dir path.")
        sys.exit(1)

    total_chunks = 0
    errors       = 0
    for pdf_path in pdfs:
        name = os.path.basename(pdf_path)
        print(f"\nProcessing: {name}")
        try:
            chunks = tp.process_file(pdf_path)
            stem   = Path(pdf_path).stem[:30]
            stem   = re.sub(r"[^\w\-]", "_", stem)
            out    = os.path.join(out_dir, f"{stem}.jsonl")
            tp.save_chunks(chunks, out)
            total_chunks += len(chunks)
            if chunks:
                print(f"  Sample: {chunks[0]['text'][:150]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Processed {len(pdfs)-errors}/{len(pdfs)} PDFs successfully")
    print(f"Total chunks: {total_chunks}")
    print(f"Errors:       {errors}")
    print(f"Output dir:   {out_dir}")
    if errors == 0:
        print("\nDay 4 complete. Text processor ready for Day 5.")