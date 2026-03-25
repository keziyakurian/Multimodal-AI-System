from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import shutil
from src.ingestion_engine import IngestionEngine
from src.vector_db import VectorDB
from src.reasoning_engine import ReasoningEngine
from src.utils.logger import get_logger

# --- INITIALIZATION ---
logger = get_logger("API_Gateway")
app = FastAPI(title="Multimodal IDP + RAG API", version="1.0.0")

logger.info("Starting Multimodal IDP API Gateway...")

# Ensure data directories exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vector_db", exist_ok=True)

# Initialize Core Services
engine = IngestionEngine()
vdb = VectorDB()
reasoner = ReasoningEngine(vdb)

# --- SCHEMAS ---
class QueryRequest(BaseModel):
    domain: str
    query: str

class IngestionResponse(BaseModel):
    filename: str
    status: str
    pages_processed: int

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal IDP API. Use /docs for documentation."}

@app.get("/health")
async def health():
    return {"status": "healthy", "engines": ["OCR", "VectorDB", "GPT-4o"]}

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(domain: str, file: UploadFile = File(...)):
    """Uploads, extracts, and stores a document in the vector database."""
    file_path = f"data/uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. OCR & Extraction
        extracted_data = engine.load_document(file_path)
        
        # 2. Vector Storage
        all_text = " ".join([page['content'] for page in extracted_data])
        vdb.add_documents(
            domain=domain,
            docs=[all_text],
            metadatas=[{"source": file.filename, "type": "api_upload"}],
            ids=[f"API-{file.filename[:5]}-{os.urandom(2).hex()}"]
        )
        
        return {
            "filename": file.filename,
            "status": "Success: Stored in Vector DB",
            "pages_processed": len(extracted_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
async def query_system(request: QueryRequest):
    """Answers questions based on retrieved knowledge from the specific domain."""
    try:
        answer = reasoner.search_and_summarize(request.domain, request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
