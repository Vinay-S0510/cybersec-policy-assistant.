from src.ingestion import build_chunks
from src.vectorstore import build_vector_store


def main():
    print("=" * 60)
    print("CYBERSECURITY POLICY & COMPLIANCE ASSISTANT")
    print("BUILD PIPELINE")
    print("=" * 60)

    print("\n[1/2] Building document chunks...")
    build_chunks()

    print("\n[2/2] Building vector store...")
    build_vector_store()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()