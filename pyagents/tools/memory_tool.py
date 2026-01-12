import os
import json
import time
from typing import List, Dict
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from rich import print
from pyagents.config import EMBEDDING_MODEL

MEMORY_FILE = os.path.expanduser("~/.pyagents/memory.json")

class MemoryManager:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.vector_store = None
        self.is_ready = False
        self._ensure_memory_file()

    def _ensure_memory_file(self):
        if not os.path.exists(os.path.dirname(self.memory_file)):
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w") as f:
                json.dump([], f)

    def load_memories(self) -> List[Dict]:
        with open(self.memory_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def save_memory(self, summary: str):
        memories = self.load_memories()
        new_memory = {
            "content": summary,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        memories.append(new_memory)
        with open(self.memory_file, "w") as f:
            json.dump(memories, f, indent=2)
        print(f"[Memory] 💾 Saved new memory: {summary[:50]}...")

    def ingest(self):
        """Builds the vector index from the memory file."""
        print(f"[Memory] 🧠 Loading long-term memories from '{self.memory_file}'...")
        memories = self.load_memories()

        if not memories:
            print("[Memory] No memories found.")
            self.is_ready = False
            return

        documents = []
        for mem in memories:
            doc = Document(
                page_content=mem["content"],
                metadata={"timestamp": mem["timestamp"], "date": mem["date"]}
            )
            documents.append(doc)

        # Using the same embedding model as RAG tool for consistency
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        try:
            self.vector_store = FAISS.from_documents(documents, embeddings)
            self.is_ready = True
            print(f"[Memory] ✅ Loaded {len(documents)} memories.")
        except Exception as e:
            print(f"[Memory] ❌ Failed to load memories: {e}")
            self.is_ready = False

    def query(self, question: str, k: int = 3) -> str:
        """Retrieves the most relevant memories."""
        if not self.is_ready:
            return "No memories available."

        docs = self.vector_store.similarity_search(question, k=k)

        context = ""
        for i, doc in enumerate(docs):
            context += f"\n--- MEMORY ({doc.metadata.get('date', 'Unknown Date')}) ---\n"
            context += doc.page_content + "\n"

        return context
