import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from rich import print

class LocalLibrarian:
    def __init__(self, target_folder: str, file_extensions: List[str] = [".py", ".lean", ".md"]):
        self.target_folder = target_folder
        self.extensions = file_extensions
        self.vector_store = None
        self.is_ready = False

    def ingest(self):
        """Scans the folder, chunks text, and builds the index."""
        print(f"[Librarian] 📚 Scanning '{self.target_folder}'...")

        documents = []
        for ext in self.extensions:
            files = glob.glob(os.path.join(self.target_folder, "**", f"*{ext}"), recursive=True)
            for file_path in files:
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

        if not documents:
            print("[Librarian] No documents found.")
            return

        # Split chunks (important for code context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # Using a default embedding model, could be configurable
        embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")

        print(f"[Librarian] Embedding {len(splits)} chunks...")
        self.vector_store = FAISS.from_documents(splits, embeddings)
        self.is_ready = True
        print("[Librarian] ✅ Index built.")

    def query(self, question: str, k: int = 4) -> str:
        """Retrieves the most relevant code snippets."""
        if not self.is_ready:
            return "Error: Librarian not initialized. Run ingest() first."

        docs = self.vector_store.similarity_search(question, k=k)

        context = ""
        for i, doc in enumerate(docs):
            context += f"\n--- SOURCE: {doc.metadata['source']} ---\n"
            context += doc.page_content + "\n"

        return context

def main():
    # Test on the current directory
    lib = LocalLibrarian("./")
    lib.ingest()
    res = lib.query("how does the quorum work?")
    print(res)

if __name__ == "__main__":
    main()
