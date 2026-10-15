from dotenv import load_dotenv

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import build_chunks
from src.retrieval.vector_store import build_vector_store


def main():
    load_dotenv()

    pages = load_all_pdfs()
    chunks = build_chunks(pages)

    print(f"\nPages  : {len(pages)}")
    print(f"Chunks : {len(chunks)}")

    build_vector_store(chunks)


if __name__ == "__main__":
    main()
