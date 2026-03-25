import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from vector_db import VectorDB
from typing import List, Dict

# Load environment variables (API Keys)
load_dotenv()

class ReasoningEngine:
    def __init__(self, vdb: VectorDB):
        self.vdb = vdb
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def search_and_summarize(self, domain: str, query: str):
        """
        Retrieves relevant documents and uses GPT-4o to provide
        a reasoning-based answer.
        """
        # 1. RETRIEVAL
        search_results = self.vdb.query_documents(domain, query, n_results=3)
        
        extracted_docs = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        
        if not extracted_docs:
            return "❌ No relevant information found in the knowledge base."

        # 2. PROMPT CONSTRUCTION
        context = "\n---\n".join(extracted_docs)
        prompt = ChatPromptTemplate.from_template("""
        You are a professional Document Analysis AI specialized in {domain}.
        Use the following retrieved context to answer the user's question.
        If the answer isn't in the context, say you don't know, don't make it up.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        ANSWER (Be concise and professional):
        """)

        # 3. GENERATION
        chain = prompt | self.llm
        try:
            response = chain.invoke({"domain": domain, "context": context, "query": query})
            
            # Format output with Sources
            source_file = metadatas[0].get('source', 'Unknown')
            final_output = f"### 🤖 AI Analysis ({domain.capitalize()})\n\n"
            final_output += response.content
            final_output += f"\n\n**Source Document:** `{source_file}`"
            
            return final_output
        except Exception as e:
            return f"Error during AI reasoning: {str(e)}"

if __name__ == "__main__":
    # Test
    from vector_db import VectorDB
    vdb = VectorDB()
    reasoner = ReasoningEngine(vdb)
    print("Reasoning Engine Initialized.")
