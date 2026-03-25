import os
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import fitz # pyMuPDF
import easyocr
import pandas as pd

# --- 1. DATA SCHEMAS (Pydantic) ---
# Ensuring strict structure for high-stakes industries

class MedicalRecord(BaseModel):
    patient_id: str = Field(..., description="Unique ID for the patient")
    visit_date: str = Field(..., description="Date of the consultation or scan")
    diagnosis: List[str] = Field(default_factory=list, description="Extracted medical conditions")
    medications: List[str] = Field(default_factory=list, description="List of prescribed drugs")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)

class BankStatement(BaseModel):
    account_number: str = Field(..., description="Last 4 digits of the account")
    statement_period: str = Field(..., description="Start and end dates")
    ending_balance: float = Field(..., description="Final balance on the statement")
    transactions: List[Dict] = Field(default_factory=list, description="Table-extracted transactions")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)

class InsuranceClaim(BaseModel):
    claim_number: str = Field(..., description="The unique insurance claim identifier")
    policy_holder: str = Field(..., description="Name on the policy")
    incident_date: str = Field(..., description="Date of the incident reported")
    claim_amount: float = Field(..., description="Total amount requested")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)

# --- 2. THE EXTRACTION ENGINE ---

class IngestionEngine:
    def __init__(self):
        # Initialize the OCR reader (supports English by default)
        self.reader = easyocr.Reader(['en'])
        
    def load_document(self, file_path: str):
        """Loads a PDF or Image and prepares it for processing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            return self._process_pdf(file_path)
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_image(self, image_path: str):
        """Extracts text and coordinates from a single image file."""
        print(f"OCR: Processing image {image_path}...")
        results = self.reader.readtext(image_path)
        
        full_text = " ".join([res[1] for res in results])
        # Calculate average confidence
        avg_confidence = sum([res[2] for res in results]) / len(results) if results else 0.0
        
        # We store metadata like bounding boxes for "Evidence Highlighting" later
        metadata = [{"text": res[1], "bbox": res[0], "confidence": res[2]} for res in results]
        
        return {"content": full_text, "metadata": metadata, "confidence": avg_confidence}

    def _process_image_as_ocr(self, page):
        """Processes a single PDF page as an image via OCR."""
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        
        print(f"OCR: Processing scanned PDF page...")
        results = self.reader.readtext(img_data)
        
        full_text = " ".join([res[1] for res in results])
        avg_confidence = sum([res[2] for res in results]) / len(results) if results else 0.0
        return full_text, avg_confidence

    def _process_pdf(self, pdf_path: str):
        """Uses pyMuPDF for digital text and EasyOCR for scans."""
        doc = fitz.open(pdf_path)
        extracted_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # If the page is empty or has very little text, it's likely a scan
            if len(text.strip()) < 50:
                print(f"Page {page_num} detected as a scan. Running OCR...")
                text = self._process_image_as_ocr(page)
            
            # Optional: Add table extraction logic here
            tables = self._extract_tables(page)
            
            extracted_data.append({
                "page": page_num, 
                "content": text,
                "tables": tables
            })
            
        return extracted_data

    def _extract_tables(self, page):
        """Placeholder for table preservation logic."""
        # In a real-world scenario, we'd use vision models or tabula-py
        return []

if __name__ == "__main__":
    # Test initialization
    engine = IngestionEngine()
    print("Ingestion Engine Initialized and OCR Ready.")
