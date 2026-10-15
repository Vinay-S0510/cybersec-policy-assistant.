from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "cybersecurity_policy"

EMBEDDING_MODEL = "text-embedding-3-small"


# Load .env
load_dotenv()


# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------

def get_openai_client() -> OpenAI:
    """Create the OpenAI client."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Check your .env file."
        )

    return OpenAI()


def get_collection():
    """Connect to the existing ChromaDB collection."""

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_DIR}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ---------------------------------------------------------
# Query embedding
# ---------------------------------------------------------

def create_query_embedding(
    client: OpenAI,
    query: str,
) -> list[float]:
    """Create an embedding for the user's question."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    return response.data[0].embedding


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a question.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    openai_client = get_openai_client()
    collection = get_collection()

    query_embedding = create_query_embedding(
        openai_client,
        query,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    retrieved = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "text": document,
                "source": metadata.get("source"),
                "source_id": metadata.get("source_id"),
                "document": metadata.get("document"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "distance": distance,
            }
        )

    return retrieved


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def print_results(
    query: str,
    results: list[dict],
) -> None:
    """Print retrieved evidence in a readable format."""

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nQuestion:\n{query}")

    for result in results:
        print("\n" + "-" * 80)

        print(
            f"Rank: {result['rank']} | "
            f"Distance: {result['distance']:.4f}"
        )

        print(f"Source : {result['source']}")
        print(f"Page   : {result['page']}")
        print(f"Section: {result['section']}")
        print(f"Chunk  : {result['chunk_id']}")

        print("\nEvidence:")
        print(result["text"])


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nEnter your cybersecurity question: "
    ).strip()

    results = retrieve(
        query=question,
        top_k=5,
    )

    print_results(
        query=question,
        results=results,
    )