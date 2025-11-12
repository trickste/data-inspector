# app/rag/chunker.py
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """Basic cleanup: remove scripts, styles, tags, and extra whitespace."""
    text = re.sub(r"<(script|style).*?>.*?</\\1>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split large text into smaller overlapping chunks for summarization or embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def prepare_chunks(raw_html: str) -> List[str]:
    """Full cleaning + chunking pipeline."""
    cleaned = clean_text(raw_html)
    chunks = chunk_text(cleaned)
    return chunks
