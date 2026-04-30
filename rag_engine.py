from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from pdf_loader import load_pdf_from_url, load_documents_from_pdf
from text_processor import setup_advanced_text_processing, create_node_parser, release_embed_model
from gemini_client import initialize_gemini_llm
import os
import hashlib
import gc


class RAGEngine:
    def __init__(self, pdf_url, storage_dir="./storage"):
        self.storage_dir = storage_dir
        self.pdf_url = pdf_url

        print("Setting up advanced text processing with state-of-the-art embeddings...")
        self.embed_model = setup_advanced_text_processing()

        print("Initializing Gemini LLM...")
        self.llm = initialize_gemini_llm()

        print("Creating node parser...")
        self.parser = create_node_parser()
        Settings.node_parser = self.parser

        index_path = self._get_index_path()

        if os.path.exists(index_path):
            print(f"Found existing embeddings for this PDF! Loading from cache...")
            print(f"Storage location: {index_path}")
            self.index = self._load_index(index_path)
            print("Index loaded successfully! Skipping embedding generation.")
        else:
            print("No cached embeddings found. Processing PDF for the first time...")
            self.index = self._build_index(pdf_url, index_path)

        print("Creating query engine with advanced retrieval...")
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
        )

        print("RAG Engine ready with state-of-the-art processing!")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, pdf_url, index_path):
        print("Loading PDF from URL...")
        pdf_path = load_pdf_from_url(pdf_url)

        print("Extracting documents from PDF...")
        documents = load_documents_from_pdf(pdf_path)
        # pdf_path already deleted inside load_documents_from_pdf

        print("Building vector index with semantic search...")
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
        )

        # Free document objects — embeddings are now inside the index
        del documents
        gc.collect()

        print("Saving embeddings to disk for future use...")
        self._save_index(index, index_path)
        print(f"Embeddings saved to: {index_path}")

        # Release embedding model from RAM/VRAM after index is built
        release_embed_model(self.embed_model)
        self.embed_model = None

        return index

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _get_index_path(self):
        url_hash = hashlib.md5(self.pdf_url.encode()).hexdigest()
        return os.path.join(self.storage_dir, f"index_{url_hash}")

    def _save_index(self, index, index_path):
        os.makedirs(index_path, exist_ok=True)
        index.storage_context.persist(persist_dir=index_path)

    def _load_index(self, index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        return load_index_from_storage(storage_context)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question):
        if not question.strip():
            return "Please enter a non-empty question."

        print(f"\nQuery: {question}")
        print("Performing semantic retrieval and generating answer...")

        try:
            response = self.query_engine.query(question)
        except Exception as e:
            return f"Error during query: {e}"

        return str(response)
