import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict

class VectorDB:
    def __init__(self, db_path: str = "data/vector_db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or Get Collections for different domains
        self.banking_collection = self.client.get_or_create_collection("banking_idp")
        self.healthcare_collection = self.client.get_or_create_collection("healthcare_idp")
        self.insurance_collection = self.client.get_or_create_collection("insurance_idp")

    def add_documents(self, domain: str, docs: List[str], metadatas: List[Dict], ids: List[str]):
        """Adds extracted text to the specified collection."""
        if domain == "banking":
            collection = self.banking_collection
        elif domain == "healthcare":
            collection = self.healthcare_collection
        else:
            collection = self.insurance_collection
            
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(docs)} documents to {domain} collection.")

    def query_documents(self, domain: str, query_text: str, n_results: int = 3):
        """Perform semantic search across the collection."""
        if domain == "banking":
            collection = self.banking_collection
        elif domain == "healthcare":
            collection = self.healthcare_collection
        else:
            collection = self.insurance_collection
            
        return collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

if __name__ == "__main__":
    # Test initialization
    db = VectorDB()
    print("Vector Database Initialized (ChromaDB).")
