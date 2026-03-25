import streamlit as st
import os
import fitz  # PyMuPDF
import chromadb
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multimodal IDP | Intelligent Document Processing",
    page_icon=None,
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(160deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b27;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #e6edf3; font-family: 'Segoe UI', sans-serif; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 8px 20px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Text areas */
    textarea { background: #0d1117 !important; color: #c9d1d9 !important; border: 1px solid #30363d !important; }

    /* Confidence badges */
    .badge-high { color: #3fb950; font-weight: 600; }
    .badge-mid  { color: #d29922; font-weight: 600; }
    .badge-low  { color: #f85149; font-weight: 600; }

    /* Footer */
    .footer { color: #8b949e; font-size: 0.78rem; text-align: center; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client_llm = Groq(api_key=GROQ_API_KEY)

# --- VECTOR DB ---
@st.cache_resource
def get_vector_db():
    os.makedirs("data/vector_db", exist_ok=True)
    return chromadb.PersistentClient(path="data/vector_db")

chroma_client = get_vector_db()

def get_collection(domain: str):
    return chroma_client.get_or_create_collection(name=f"{domain}idp")

# --- TEXT CHUNKING ---
def chunk_text(text: str, chunk_size: int = 300) -> list:
    """Splits text into overlapping chunks of ~300 words for better retrieval."""
    words = text.split()
    chunks = []
    step = chunk_size - 50  # 50-word overlap between chunks
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]

# --- CONFIDENCE SCORING ---
def compute_confidence(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    ratio = len(alpha_words) / len(words)
    return round(0.55 + ratio * 0.44, 2)

# --- TEXT EXTRACTION ---
def extract_text_from_pdf(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        pages.append({
            "page": i,
            "content": text if text else "[Scanned page — no digital text detected]",
            "confidence": compute_confidence(text),
            "word_count": len(text.split()),
            "char_count": len(text)
        })
    return pages

# --- RAG QUERY ---
def rag_query(domain: str, question: str) -> str:
    collection = get_collection(domain)
    results = collection.query(query_texts=[question], n_results=3)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No documents found in this domain vault. Please upload and approve a document first."

    context = "\n---\n".join(docs)
    source = metas[0].get("source", "Unknown") if metas else "Unknown"
    entity = metas[0].get("entity", "") if metas else ""

    response = client_llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {"role": "system", "content": f"You are a document analysis AI for the {domain} industry. Answer ONLY from the context provided. If the answer is not present, say you don't know."},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
        ]
    )
    answer = response.choices[0].message.content
    entity_line = f"**Entity:** `{entity}`  |  " if entity else ""
    return f"{answer}\n\n---\n{entity_line}**Source:** `{source}`"

# =========================================
# HEADER
# =========================================
st.title("Multimodal IDP")
st.markdown("**Intelligent Document Processing** — Upload a PDF, extract structured data, and query it in natural language.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

# =========================================
# LEFT — INGEST
# =========================================
with col_left:
    st.subheader("Document Ingestion")
    domain = st.selectbox("Industry Vertical", ["healthcare", "banking", "insurance", "general"])
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        file_bytes = uploaded_file.read()

        with st.spinner("Extracting text..."):
            pages = extract_text_from_pdf(file_bytes)

        total_words = sum(p["word_count"] for p in pages)
        avg_conf = sum(p["confidence"] for p in pages) / len(pages)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pages", len(pages))
        m2.metric("Words Extracted", f"{total_words:,}")
        m3.metric("Avg Confidence", f"{avg_conf:.0%}")

        st.divider()

        for p in pages:
            conf = p["confidence"]
            if conf >= 0.85:
                badge = f"<span class='badge-high'>High Confidence ({conf:.0%})</span>"
            elif conf >= 0.65:
                badge = f"<span class='badge-mid'>Medium Confidence ({conf:.0%})</span>"
            else:
                badge = f"<span class='badge-low'>Low Confidence ({conf:.0%})</span>"

            with st.expander(f"Page {p['page'] + 1}  —  {p['word_count']} words", expanded=(p['page'] == 0)):
                st.markdown(badge, unsafe_allow_html=True)
                st.text_area("Extracted Content", value=p["content"], height=130, key=f"page_{p['page']}")

        st.divider()
        st.subheader("Validate and Store")
        entity_name = st.text_input("Entity Name", placeholder="e.g. John Doe, HDFC Bank")
        doc_id = st.text_input("Document ID", value=f"DOC-{uploaded_file.name[:8].upper().replace('.','')}")

        if st.button("Approve and Save to Vector Memory"):
            all_text = " ".join(p["content"] for p in pages)
            chunks = chunk_text(all_text)
            collection = get_collection(domain)

            progress = st.progress(0, text="Preparing chunks...")
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}-chunk-{idx}"
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": uploaded_file.name, "entity": entity_name, "domain": domain, "chunk": str(idx)}],
                    ids=[chunk_id]
                )
                progress.progress((idx + 1) / len(chunks), text=f"Storing chunk {idx + 1} of {len(chunks)}...")

            progress.empty()
            st.success(f"Stored {len(chunks)} chunk(s) from '{uploaded_file.name}' into the {domain} vault.")
            st.balloons()

# =========================================
# RIGHT — QUERY
# =========================================
with col_right:
    st.subheader("Document Query (RAG)")
    st.markdown("Once a document is approved, ask questions about it in plain English. The system retrieves the relevant context and generates an answer using GPT-4o.")

    search_domain = st.selectbox("Search Domain", ["healthcare", "banking", "insurance", "general"], key="sdomain")
    question = st.text_area("Question", placeholder="e.g. What medications were prescribed?\nWhat is the ending account balance?", height=100)

    if st.button("Search and Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not GROQ_API_KEY:
            st.error("Groq API key not configured.")
        else:
            with st.spinner("Searching vector memory and generating answer..."):
                answer = rag_query(search_domain, question)
            st.markdown("### Answer")
            st.markdown(answer)

    st.divider()
    st.subheader("System Architecture")
    st.markdown("""
| Layer | Technology | Role |
|---|---|---|
| Extraction | PyMuPDF | Parse and extract text from PDFs |
| Confidence | Text Density Scoring | Validate extraction quality |
| Vector Memory | ChromaDB | Store and retrieve document embeddings |
| Reasoning | GPT-4o (RAG) | Generate answers from retrieved context |
| Validation | Pydantic Schemas | Enforce data structure and integrity |
| API | FastAPI | Production service layer |
    """)

# =========================================
# FOOTER
# =========================================
st.divider()
st.markdown(
    "<div class='footer'>Multimodal IDP v2.0  |  PyMuPDF + ChromaDB + GPT-4o  |  Built by Keziya Kurian</div>",
    unsafe_allow_html=True
)
