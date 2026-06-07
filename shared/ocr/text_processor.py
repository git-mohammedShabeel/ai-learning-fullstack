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
      "page":     2           # -1 for images
    }
"""

import os
import re
import json
import glob
import unicodedata
from pathlib import Path


class TextProcessor:

    def __init__(self,
                 chunk_size: int = 400,
                 chunk_overlap: int = 50):
        """
        chunk_size    : target token count per chunk (words approx)
        chunk_overlap : overlap between consecutive chunks so context
                        is not lost at boundaries
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def process(self, raw_text: str,
                source: str = "unknown",
                page: int = -1) -> list[dict]:
        """Clean text and return list of chunk dicts."""
        cleaned = self._clean(raw_text)
        if not cleaned:
            return []
        chunks  = self._chunk(cleaned)
        return self._label(chunks, source, page)

    def process_file(self, file_path: str) -> list[dict]:
        """
        Full pipeline: OCR -> clean -> chunk.
        Imports OCREngine at call time so this module stays standalone.
        """
        # lazy import so text_processor works without easyocr if needed
        try:
            from ocr_engine import OCREngine
        except ImportError:
            raise ImportError("ocr_engine.py must be in the same folder.")

        ocr    = OCREngine()
        result = ocr.extract(file_path)
        return self.process(result["text"],
                            source=os.path.basename(file_path),
                            page=result["pages"])

    def save_chunks(self, chunks: list[dict], out_path: str):
        """Save chunks to a JSONL file (one JSON object per line)."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"  Saved {len(chunks)} chunks -> {out_path}")

    def load_chunks(self, jsonl_path: str) -> list[dict]:
        """Load chunks back from a JSONL file."""
        chunks = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    # ------------------------------------------------------------------
    # PRIVATE — cleaning
    # ------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        # Normalise unicode (fix garbled chars from OCR)
        text = unicodedata.normalize("NFKC", text)

        # Remove null bytes and control chars (except newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Fix common OCR artefacts
        text = re.sub(r"ﬁ", "fi", text)
        text = re.sub(r"ﬂ", "fl", text)
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r"[""]", '"', text)

        # Remove header/footer noise (page numbers, running titles)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*Page \d+.*$", "", text, flags=re.MULTILINE|re.IGNORECASE)

        # Collapse excessive whitespace / blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lines that are just special characters (table borders etc.)
        text = re.sub(r"^[\s\-=_|+]{5,}$", "", text, flags=re.MULTILINE)

        return text.strip()

    # ------------------------------------------------------------------
    # PRIVATE — chunking (sliding window over sentences)
    # ------------------------------------------------------------------
    def _chunk(self, text: str) -> list[str]:
        # Split into sentences using punctuation boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks   = []
        current  = []
        cur_len  = 0

        for sentence in sentences:
            words = sentence.split()
            wlen  = len(words)

            if cur_len + wlen > self.chunk_size and current:
                # Save current chunk
                chunks.append(" ".join(current))
                # Keep overlap — last N words carry over
                overlap_words = " ".join(current).split()[-self.chunk_overlap:]
                current  = list(overlap_words)
                cur_len  = len(overlap_words)

            current.append(sentence)
            cur_len += wlen

        if current:
            chunks.append(" ".join(current))

        # Filter out very short chunks (likely noise)
        chunks = [c for c in chunks if len(c.split()) >= 20]
        return chunks

    # ------------------------------------------------------------------
    # PRIVATE — labelling
    # ------------------------------------------------------------------
    def _label(self, chunks: list[str],
               source: str, page: int) -> list[dict]:
        stem = Path(source).stem[:20]
        return [
            {
                "id":     f"{stem}_chunk_{i:04d}",
                "text":   chunk,
                "tokens": len(chunk.split()),
                "source": source,
                "page":   page
            }
            for i, chunk in enumerate(chunks)
        ]


# ------------------------------------------------------------------
# Self-test — python text_processor.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    pdf_dir   = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-data", "pdfs")
    out_dir   = os.path.join(os.path.expanduser("~"), "Desktop", "DL","ai-learning-fullstack", "shared", "processed")
    os.makedirs(out_dir, exist_ok=True)

    tp   = TextProcessor(chunk_size=400, chunk_overlap=50)
    pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))[:5]

    if not pdfs:
        print("No PDFs found. Check your pdf_dir path.")
        exit(1)

    total_chunks = 0
    for pdf_path in pdfs:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        try:
            chunks = tp.process_file(pdf_path)
            name   = Path(pdf_path).stem[:30]
            out    = os.path.join(out_dir, f"{name}.jsonl")
            tp.save_chunks(chunks, out)
            total_chunks += len(chunks)

            # Print first chunk as sample
            if chunks:
                print(f"  Sample chunk [{chunks[0]['id']}]:")
                print(f"  {chunks[0]['text'][:200]}...")
        except Exception as e:
            print(f"  ERROR on {os.path.basename(pdf_path)}: {e}")

    print(f"\n{'='*50}")
    print(f"Processed {len(pdfs)} PDFs -> {total_chunks} total chunks")
    print(f"Chunks saved to: {out_dir}")
    print("\nDay 4 complete. Text processor ready.")