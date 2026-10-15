from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "chunks.json"


# -------------------------------------------------------------------
# Document configuration
# -------------------------------------------------------------------

DOCUMENTS = {
    "CIS_Controls__v8__Critical_Security_Controls__2023_08.pdf": {
        "source": "CIS Controls v8",
        "source_id": "cis_v8",
    },
    "NIST.CSWP.29.pdf": {
        "source": "NIST Cybersecurity Framework 2.0",
        "source_id": "nist_csf_2",
    },
    "NIST.SP.800-53r5.pdf": {
        "source": "NIST SP 800-53 Revision 5",
        "source_id": "nist_800_53_r5",
    },
}


# -------------------------------------------------------------------
# Text cleaning
# -------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split across lines with a hyphen.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace multiple spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace around lines.
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


# -------------------------------------------------------------------
# Heading detection
# -------------------------------------------------------------------

def looks_like_heading(line: str) -> bool:
    """
    Heuristic heading detector.

    We deliberately keep this conservative because different PDFs
    have different formatting structures.
    """

    line = line.strip()

    if not line:
        return False

    if len(line) > 180:
        return False

    # Common numbered headings.
    if re.match(r"^\d+(\.\d+)*[\s.)-]+", line):
        return True

    # Control identifiers such as AC-1, IA-2, etc.
    if re.match(r"^[A-Z]{2,5}-\d+(\s|:|-)", line):
        return True

    # Fully uppercase short lines.
    letters = re.sub(r"[^A-Za-z]", "", line)

    if len(letters) >= 4 and line.upper() == line and len(line) <= 120:
        return True

    return False


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------

def split_into_chunks(
    text: str,
    max_chars: int = 3500,
    overlap: int = 500,
) -> list[str]:
    """
    Split text into reasonably sized chunks while preserving paragraph
    boundaries where possible.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # If the paragraph itself is larger than the limit,
        # split it by sentences/lines.
        if len(paragraph) > max_chars:
            pieces = re.split(r"(?<=[.!?])\s+", paragraph)

            for piece in pieces:
                piece = piece.strip()

                if not piece:
                    continue

                if len(current) + len(piece) + 1 <= max_chars:
                    current = f"{current} {piece}".strip()
                else:
                    if current:
                        chunks.append(current)

                    # Preserve overlap from the previous chunk.
                    previous_tail = (
                        current[-overlap:]
                        if current
                        else ""
                    )

                    current = f"{previous_tail} {piece}".strip()

        else:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)

                previous_tail = (
                    current[-overlap:]
                    if current
                    else ""
                )

                current = f"{previous_tail}\n\n{paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


# -------------------------------------------------------------------
# PDF processing
# -------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    source: str,
    source_id: str,
    start_chunk_id: int,
) -> tuple[list[dict], int]:
    """Extract and chunk a single PDF."""

    print(f"\nProcessing: {source}")
    print(f"File: {pdf_path.name}")

    document = pymupdf.open(pdf_path)

    chunks: list[dict] = []
    chunk_id = start_chunk_id

    current_section = "Unknown"

    print(f"Pages: {len(document)}")

    for page_index, page in enumerate(document):
        page_number = page_index + 1

        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        if not text:
            continue

        # Track likely section headings on the page.
        for line in text.splitlines():
            if looks_like_heading(line):
                current_section = line

        page_chunks = split_into_chunks(text)

        for local_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id:05d}",
                    "source": source,
                    "source_id": source_id,
                    "document": pdf_path.name,
                    "page": page_number,
                    "section": current_section,
                    "chunk_index": local_index,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    document.close()

    print(f"Chunks created: {len(chunks)}")

    return chunks, chunk_id


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def build_chunks() -> list[dict]:
    """Process all configured documents."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    next_chunk_id = 1

    for filename, metadata in DOCUMENTS.items():
        pdf_path = RAW_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Required PDF not found: {pdf_path}"
            )

        chunks, next_chunk_id = process_pdf(
            pdf_path=pdf_path,
            source=metadata["source"],
            source_id=metadata["source_id"],
            start_chunk_id=next_chunk_id,
        )

        all_chunks.extend(chunks)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total chunks : {len(all_chunks)}")
    print(f"Output      : {OUTPUT_FILE}")

    # Summary by source.
    source_counts: dict[str, int] = {}

    for chunk in all_chunks:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nChunks by source:")

    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    return all_chunks


if __name__ == "__main__":
    build_chunks()