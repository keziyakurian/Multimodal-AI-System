import streamlit as st
import os
import fitz  # PyMuPDF
import chromadb
from groq import Groq
import uuid
import sys
import requests
from streamlit_mic_recorder import mic_recorder
from src.voice_engine import VoiceEngine, get_audio_html
from src.agent_engine import AgenticEngine

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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

# --- TEXT EXTRACTION (Deprecated - Now routed to Microservice) ---
# See microservice_app.py for the full GPU / OCR extraction logic

# --- RAG QUERY ---
def rag_query(domain: str, question: str) -> str:
    collection = get_collection(domain)
    # Enable Session-based Metadata Filtering!
    results = collection.query(
        query_texts=[question], 
        n_results=3,
        where={"session_id": st.session_state.session_id}
    )
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
    pipeline = st.radio("Ingestion Pipeline", ["Standard (Fast CPU OCR)", "Scientific / Equation Heavy (Surya Marker GPU)"])
    
    # Enable multiple PDFs and images
    uploaded_files = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        pipeline_val = "scientific" if "Scientific" in pipeline else "standard"
        all_pages = []

        if pipeline_val == "scientific":
            with st.spinner(f"Routing to {pipeline_val.upper()} Microservice for heavy GPU extraction..."):
                import requests
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    try:
                        files = {"file": (uploaded_file.name, file_bytes, "application/octet-stream")}
                        data = {"pipeline_type": pipeline_val}
                        res = requests.post("https://multimodal-ai-system.onrender.com/extract", files=files, data=data)
                        
                        if res.status_code == 200 and res.json().get("status") == "success":
                            pages = res.json()["pages"]
                            for page in pages:
                                page["source_file"] = uploaded_file.name
                            all_pages.extend(pages)
                        else:
                            st.error(f"Microservice Error for {uploaded_file.name}: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to Ingestion Microservice. Ensure it's running or wait for Render Cold Start.")
        else:
            with st.spinner("Extracting standard digital text locally (Instant)..."):
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    file_extension = uploaded_file.name.split('.')[-1].lower() if uploaded_file.name else "pdf"
                    
                    try:
                        doc = fitz.open(stream=file_bytes, filetype=file_extension)
                        for i, page in enumerate(doc):
                            text = page.get_text().strip()
                            all_pages.append({
                                "source_file": uploaded_file.name,
                                "page": i,
                                "content": text if text else "[Scanned page — no digital text detected]",
                                "confidence": compute_confidence(text),
                                "word_count": len(text.split()),
                                "char_count": len(text)
                            })
                    except Exception as e:
                        st.error(f"Could not parse {uploaded_file.name} locally: {e}")

        if all_pages:
            total_words = sum(p["word_count"] for p in all_pages)
            avg_conf = sum(p["confidence"] for p in all_pages) / len(all_pages) if all_pages else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Pages extracted", len(all_pages))
            m2.metric("Words Extracted", f"{total_words:,}")
            m3.metric("Avg Confidence", f"{avg_conf:.0%}")

        st.divider()

        for p in all_pages:
            conf = p["confidence"]
            if conf >= 0.85:
                badge = f"<span class='badge-high'>High Confidence ({conf:.0%})</span>"
            elif conf >= 0.65:
                badge = f"<span class='badge-mid'>Medium Confidence ({conf:.0%})</span>"
            else:
                badge = f"<span class='badge-low'>Low Confidence ({conf:.0%})</span>"

            source_name = p.get("source_file", "Unknown")
            with st.expander(f"{source_name} — Page {p['page'] + 1}  ({p['word_count']} words)", expanded=False):
                st.markdown(badge, unsafe_allow_html=True)
                st.text_area("Extracted Content", value=p["content"], height=130, key=f"page_{source_name}_{p['page']}")

        st.divider()
        st.subheader("Validate and Store")
        entity_name = st.text_input("Entity Name", placeholder="e.g. John Doe, HDFC Bank")
        
        default_batch_id = f"BATCH-{uploaded_files[0].name[:8].upper().replace('.','')}" if uploaded_files else "BATCH-01"
        doc_id = st.text_input("Document/Batch ID", value=default_batch_id)

        if st.button("Approve and Save to Vector Memory") and all_pages:
            all_text = " ".join(p["content"] for p in all_pages)
            chunks = chunk_text(all_text)
            collection = get_collection(domain)

            progress = st.progress(0, text="Preparing chunks...")
            
            source_files_str = ", ".join([f.name for f in uploaded_files])
            
            for idx, chunk in enumerate(chunks):
                # Unique ID locked to this specific user session
                chunk_id = f"{st.session_state.session_id}-{doc_id}-chunk-{idx}"
                collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "source": source_files_str, 
                        "entity": entity_name, 
                        "domain": domain, 
                        "chunk": str(idx),
                        "session_id": st.session_state.session_id  # <--- METADATA FILTER TAG
                    }],
                    ids=[chunk_id]
                )
                progress.progress((idx + 1) / len(chunks), text=f"Storing chunk {idx + 1} of {len(chunks)}...")

            progress.empty()
            st.success(f"Stored {len(chunks)} chunk(s) securely into the {domain} vault (Isolated to your session window).")
            st.balloons()

# =========================================
# RIGHT — QUERY (Agentic Voice Assistant)
# =========================================
with col_right:
    st.subheader("Agentic Voice Assistant")
    st.markdown("Ask complex questions or give commands. The agent uses tools to reason across documents.")

    # Voice Engine & Agent Init
    voice_engine = VoiceEngine()
    
    def agent_rag_search(query: str):
        # This is the tool function for the agent
        return rag_query(st.session_state.get("sdomain", "general"), query)

    agent_engine = AgenticEngine(vector_db_query_fn=agent_rag_search)

    search_domain = st.selectbox("Search Domain", ["healthcare", "banking", "insurance", "general"], key="sdomain")
    
    # --- VOICE INPUT ---
    st.write("🎙️ **Voice Command**")
    audio_record = mic_recorder(
        start_prompt="Click to Speak",
        stop_prompt="Stop Recording",
        key='mic_recorder'
    )

    question = st.text_area("Question", placeholder="e.g. Based on this invoice, draft an email asking for a 10% discount.", height=100, key="agent_question")

    # If voice is captured, overwrite question
    if audio_record:
        with st.spinner("Transcribing..."):
            voice_text = voice_engine.stt(audio_record['bytes'])
            if "Error" not in voice_text:
                st.session_state.agent_question = voice_text
                st.info(f"Captured: {voice_text}")
            else:
                st.error(voice_text)

    if st.button("Execute Agentic Task"):
        input_text = st.session_state.agent_question if st.session_state.agent_question else question
        if not input_text.strip():
            st.warning("Please enter a question or speak.")
        elif not GROQ_API_KEY:
            st.error("Groq API key not configured.")
        else:
            with st.spinner("Agent is reasoning and executing tools..."):
                answer = agent_engine.run(input_text)
            
            st.markdown("### Assistant Response")
            st.markdown(answer)
            
            # --- VOICE OUTPUT ---
            if st.checkbox("Read response aloud (TTS)"):
                with st.spinner("Generating voice..."):
                    audio_bytes = voice_engine.tts(answer)
                    if audio_bytes:
                        st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                    else:
                        st.warning("TTS API Key missing or error occurred.")

    st.divider()
    st.subheader("System Architecture")
    st.markdown("""
| Layer | Technology | Role |
|---|---|---|
| Ingestion | PyMuPDF / Surya | Extract text from documents |
| STT | Deepgram (Nova-2) | Voice-to-text input |
| Agentic Layer | LangChain Agent | Tool-based reasoning loop |
| Vector Memory | ChromaDB | Isolated session-based storage |
| Reasoning | Llama-3.1 (Groq) | High-speed LLM inference |
| TTS | Cartesia (Sonic) | Text-to-speech feedback |
    """)

# =========================================
# FOOTER
# =========================================
st.divider()
st.markdown(
    "<div class='footer'>Multimodal IDP v2.0  |  PyMuPDF + ChromaDB + Llama-3.1-8b-instant  |  Built by Keziya Kurian</div>",
    unsafe_allow_html=True
)
