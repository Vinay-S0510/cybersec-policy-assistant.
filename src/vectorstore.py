from __future__ import annotations

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

load_dotenv()


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

COLLECTION_NAME = "cybersecurity_policy"

EMBEDDING_MODEL = "text-embedding-3-small"

BATCH_SIZE = 100


# -------------------------------------------------------------------
# Clients
# -------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    """Create the OpenAI client using OPENAI_API_KEY."""

    return OpenAI()


def get_chroma_collection():
    """Create/open the persistent Chroma collection."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Cybersecurity policy and compliance "
                "documents including CIS Controls v8, "
                "NIST CSF 2.0, and NIST SP 800-53 Rev. 5"
            )
        },
    )

    return collection


# -------------------------------------------------------------------
# Embedding
# -------------------------------------------------------------------

def create_embeddings(
    client: OpenAI,
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for a batch of text."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


# -------------------------------------------------------------------
# Build vector database
# -------------------------------------------------------------------

def build_vector_store() -> None:
    """Load chunks, embed them, and store them in Chroma."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run `python -m src.ingestion` first."
        )

    print("Loading chunks...")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    print(f"Chunks loaded: {len(chunks)}")

    openai_client = get_openai_client()
    collection = get_chroma_collection()

    # Rebuild the collection from scratch.
    # This prevents duplicate embeddings when build.py is run again.
    existing_count = collection.count()

    if existing_count > 0:
        print(
            f"Existing vectors found: {existing_count}"
        )
        print("Clearing existing collection...")

        # Chroma allows deleting all records by their IDs.
        existing = collection.get()

        if existing["ids"]:
            collection.delete(
                ids=existing["ids"]
            )

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"Embedding chunks "
            f"{start + 1}-{min(start + BATCH_SIZE, total)} "
            f"of {total}..."
        )

        embeddings = create_embeddings(
            openai_client,
            texts,
        )

        ids = [
            chunk["chunk_id"]
            for chunk in batch
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "source_id": chunk["source_id"],
                "document": chunk["document"],
                "page": chunk["page"],
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print("\n" + "=" * 60)
    print("VECTOR STORE COMPLETE")
    print("=" * 60)

    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vectors    : {collection.count()}")
    print(f"Location   : {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()