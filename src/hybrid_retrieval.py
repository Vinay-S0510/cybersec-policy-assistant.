from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "cybersecurity_policy"
EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 10

load_dotenv()


# ---------------------------------------------------------
# Load documents for keyword search
# ---------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text while retaining useful identifiers."""

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*",
        text.lower(),
    )


def load_chunks() -> list[dict]:
    """Load the same chunks used to build ChromaDB."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    return chunks


# ---------------------------------------------------------
# Clients and collection
# ---------------------------------------------------------

def get_openai_client() -> OpenAI:
    """Create the OpenAI client."""

    return OpenAI()


def get_collection():
    """Connect to the existing ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ---------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------

class HybridRetriever:
    """Combine semantic and keyword retrieval using RRF."""

    def __init__(self):
        print("Loading chunks for keyword search...")

        self.chunks = load_chunks()

        self.chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        print(f"Keyword-search documents: {len(self.chunks)}")

        print("Building BM25 index...")

        tokenized_documents = [
            tokenize(
                chunk["source"]
                + " "
                + chunk["document"]
                + " "
                + chunk["section"]
                + " "
                + chunk["text"]
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print("Connecting to ChromaDB...")

        self.collection = get_collection()

        print(
            f"ChromaDB vectors: {self.collection.count()}"
        )

        print("Initializing OpenAI client...")

        self.openai_client = get_openai_client()

        print("Hybrid retrieval is ready.")

    def semantic_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Retrieve semantically similar chunks."""

        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )

        query_embedding = response.data[0].embedding

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ranked_results = []

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            ranked_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "source": metadata.get("source"),
                    "source_id": metadata.get("source_id"),
                    "document": metadata.get("document"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "distance": distance,
                    "semantic_rank": rank,
                }
            )

        return ranked_results

    def keyword_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Retrieve chunks using BM25 keyword relevance."""

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        ranked_results = []

        for index in ranked_indices[:limit]:
            if scores[index] <= 0:
                continue

            chunk = self.chunks[index]

            ranked_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "source_id": chunk["source_id"],
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "keyword_score": float(scores[index]),
                }
            )

        return ranked_results

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list[list[dict]],
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Combine ranked lists using Reciprocal Rank Fusion.

        Each result contributes 1 / (rrf_k + rank).
        """

        combined = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                if chunk_id not in combined:
                    combined[chunk_id] = {
                        **result,
                        "rrf_score": 0.0,
                        "semantic_rank": None,
                        "keyword_rank": None,
                    }

                combined[chunk_id]["rrf_score"] += (
                    1.0 / (rrf_k + rank)
                )

                if "distance" in result:
                    combined[chunk_id]["distance"] = (
                        result["distance"]
                    )
                    combined[chunk_id]["semantic_rank"] = rank

                if "keyword_score" in result:
                    combined[chunk_id]["keyword_score"] = (
                        result["keyword_score"]
                    )
                    combined[chunk_id]["keyword_rank"] = rank

        return sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Perform semantic + keyword search, then fuse results."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        semantic_results = self.semantic_search(
            query,
            RETRIEVAL_CANDIDATES,
        )

        keyword_results = self.keyword_search(
            query,
            RETRIEVAL_CANDIDATES,
        )

        fused_results = self.reciprocal_rank_fusion(
            [
                semantic_results,
                keyword_results,
            ]
        )

        return fused_results[:top_k]


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

def print_results(
    query: str,
    results: list[dict],
) -> None:
    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nQuestion: {query}")

    for result in results:
        print("\n" + "-" * 80)

        print(f"Rank         : {results.index(result) + 1}")
        print(f"RRF score    : {result['rrf_score']:.6f}")
        print(f"Source       : {result['source']}")
        print(f"PDF page     : {result['page']}")
        print(f"Section      : {result['section']}")
        print(f"Chunk        : {result['chunk_id']}")
        print(f"Semantic rank: {result.get('semantic_rank')}")
        print(f"Keyword rank : {result.get('keyword_rank')}")

        if "distance" in result:
            print(f"Distance     : {result['distance']:.4f}")

        print("\nEvidence:")
        print(result["text"])


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":
    retriever = HybridRetriever()

    while True:
        question = input(
            "\nEnter a cybersecurity question "
            "(or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            results = retriever.retrieve(
                query=question,
                top_k=DEFAULT_TOP_K,
            )

            print_results(question, results)

        except Exception as error:
            print(f"\nRetrieval error: {error}")
