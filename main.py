from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import build_chunks


def main():
    pages = load_all_pdfs()
    chunks = build_chunks(pages)

    print("\n" + "=" * 60)
    print("CORPUS INGESTION COMPLETE")
    print("=" * 60)
    print(f"Pages extracted : {len(pages)}")
    print(f"Chunks created  : {len(chunks)}")

    sources = {}
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        sources[source] = sources.get(source, 0) + 1

    print("\nChunks by source:")
    for source, count in sources.items():
        print(f"  {source:15} {count}")

    print("\nSample chunks:")
    for chunk in chunks[:3]:
        print("\n---", chunk["id"], "---")
        print(chunk["text"][:700])


if __name__ == "__main__":
    main()
