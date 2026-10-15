import json
import os
from pathlib import Path

import chromadb
from openai import OpenAI


COLLECTION_NAME = "cybersecurity_policies"
DB_PATH = "data/chroma"


def create_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    # OpenAI returns embeddings in the same order as the input.
    return [item.embedding for item in response.data]


def build_vector_store(chunks: list[dict], batch_size: int = 100):
    """
    Create/update the local Chroma collection using OpenAI embeddings.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env before building the vector store."
        )

    Path(DB_PATH).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Cybersecurity policy and compliance corpus"},
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]

        # Chroma metadata values must be scalar types.
        metadatas = []
        for item in batch:
            md = {}
            for key, value in item["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    md[key] = value
            metadatas.append(md)

        embeddings = create_embeddings(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    print("\nVector store ready.")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Documents  : {collection.count()}")
    print(f"Database   : {DB_PATH}")

    return collection


def test_retrieval(query: str, n_results: int = 5):
    """
    Retrieve the most semantically relevant chunks for a test query.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for i in range(len(results["ids"][0])):
        print(f"\n[{i + 1}]")
        print("ID       :", results["ids"][0][i])
        print("Distance :", round(results["distances"][0][i], 4))
        print("Metadata :", results["metadatas"][0][i])
        print("Text     :", results["documents"][0][i][:1000])

    return results
