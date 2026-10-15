from src.retrieval import retrieve
from src.generator import generate_answer


def main():
    print("=" * 65)
    print("CYBERSECURITY POLICY & COMPLIANCE ASSISTANT")
    print("Evidence-grounded RAG")
    print("Type 'exit' to stop.")
    print("=" * 65)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            evidence = retrieve(question, top_k=5)
            answer = generate_answer(question, evidence)

            print("\nAssistant:\n")
            print(answer)

            print("\nRetrieved sources:")
            seen = set()

            for item in evidence:
                key = (
                    item.get("source"),
                    item.get("page"),
                )

                if key in seen:
                    continue

                seen.add(key)

                print(
                    f"- {item.get('source')} "
                    f"(PDF page {item.get('page')})"
                )

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
