import requests
from llama_index.readers.file import PDFReader
from pathlib import Path
import tempfile
import os
import gc

DOWNLOAD_CHUNK_SIZE = 8192  # 8 KB chunks to avoid loading entire PDF into RAM

def load_pdf_from_url(url, timeout=60):
    print(f"Downloading PDF from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timed out after {timeout}s: {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error downloading PDF: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download PDF: {e}")

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        print(f"Warning: Content-Type '{content_type}' may not be a PDF.")

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_rag_document.pdf")

    # Stream to disk in chunks — never accumulates the full PDF in RAM
    bytes_written = 0
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    print(f"Downloaded {bytes_written / 1024:.1f} KB to {temp_path}")
    return temp_path


def load_documents_from_pdf(pdf_path):
    loader = PDFReader()
    try:
        documents = loader.load_data(file=Path(pdf_path))
    finally:
        # Remove temp file immediately after extraction to free disk space
        try:
            os.remove(pdf_path)
        except OSError:
            pass
        gc.collect()

    if not documents:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    total_chars = sum(len(d.text) for d in documents)
    print(f"Extracted {len(documents)} page(s), {total_chars:,} characters total.")
    return documents
