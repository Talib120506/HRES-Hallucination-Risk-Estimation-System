"""
PDF and Text Utilities
Functions for PDF text extraction, cleaning, and chunking
"""
import re
import fitz  # PyMuPDF

# Constants
MAX_CTX_CHARS = 1600
CHUNK_SIZE = 100
CHUNK_OVERLAP = 30


def extract_all_text(pdf_path):
    """Extract text from all pages of a PDF"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 30:
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def extract_page_text(pdf_path, page_num):
    """Extract text from a specific page"""
    doc = fitz.open(pdf_path)
    idx = page_num - 1
    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""
    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CTX_CHARS]


def clean_text(text):
    """Clean extracted PDF text by removing page numbers, figures, etc."""
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip standalone page numbers
        if re.fullmatch(r'\d{1,3}', line):
            continue
        # Skip common PDF header/footer patterns
        if re.match(r'^(en|de|fr|es|it)\s+\S', line) and len(line) < 40:
            continue
        # Skip figure captions
        if re.match(r'^\(?(figure|fig\.?)\s', line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def chunk_text(text):
    """Split text into overlapping chunks"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        current.extend(sentence.split())
        if len(current) >= CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP:]
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]
