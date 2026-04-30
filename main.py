from rag_engine import RAGEngine
import sys


def main():
    pdf_url = "https://arxiv.org/pdf/2005.11401.pdf"

    try:
        rag = RAGEngine(pdf_url)
    except Exception as e:
        print(f"Failed to initialise RAG engine: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n\nInteractive mode - Enter your questions (type 'q' to exit):")
    while True:
        try:
            user_query = input("\nYour question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_query.lower() in ["q", "quit", "exit"]:
            break

        if not user_query.strip():
            continue

        answer = rag.query(user_query)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
