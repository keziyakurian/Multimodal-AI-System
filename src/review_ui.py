import streamlit as st
import os
import sys

# Ensure we can import from the src directory
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from ingestion_engine import IngestionEngine
    from vector_db import VectorDB
except ImportError:
    st.error("Could not find source modules. Make sure you are running streamlit from the project root.")

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Multimodal IDP Review Room", layout="wide")

st.title("🛡️ Multimodal IDP: Human-in-the-Loop Review")
st.markdown("""
Welcome to the Review Room. Here you can verify AI-extracted data from Insurance, Banking, and Healthcare documents.
""")

# Initialize Session State
if 'engine' not in st.session_state:
    st.session_state['engine'] = IngestionEngine()
if 'vdb' not in st.session_state:
    st.session_state['vdb'] = VectorDB()

# --- SIDEBAR: Upload & Engine ---
st.sidebar.header("📁 Document Ingestion")
domain = st.sidebar.selectbox("Industry Vertical", ["banking", "healthcare", "insurance"])
uploaded_file = st.sidebar.file_uploader("Upload a Scan or PDF", type=["pdf", "png", "jpg", "jpeg"])

# --- MAIN DASHBOARD ---
if uploaded_file is not None:
    # Save file temporarily
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Original Document")
        if uploaded_file.type == "application/pdf":
            st.info("PDF Preview: " + uploaded_file.name)
        else:
            st.image(uploaded_file, use_column_width=True)

    with col2:
        st.subheader("🔍 Extraction Results")
        
        with st.spinner("Extracting..."):
            results = st.session_state['engine'].load_document(temp_path)
            
            all_text = ""
            for page_data in results:
                st.markdown(f"### Page {page_data['page'] + 1}")
                all_text += page_data['content'] + " "
                
                conf = page_data.get('confidence', 0.85)
                if conf > 0.8:
                    st.success(f"High Confidence: {conf:.2%}")
                elif conf > 0.5:
                    st.warning(f"Medium Confidence: {conf:.2%}")
                else:
                    st.error(f"Low Confidence: {conf:.2%}")

                st.text_area(f"Text Content (Page {page_data['page'] + 1})", value=page_data['content'], height=150, key=f"text_{page_data['page']}")
            
            st.divider()
            st.subheader("📝 Verify Structured Data")
            client_name = st.text_input("Entity Name", value="Auto-Extracting...")
            doc_id = st.text_input("Document ID", value=f"DOC-{uploaded_file.name[:5].upper()}")
            
            if st.button("✅ Approve & Save to Vector DB"):
                with st.spinner("Embedding and Storing..."):
                    st.session_state['vdb'].add_documents(
                        domain=domain,
                        docs=[all_text],
                        metadatas=[{"source": uploaded_file.name, "entity": client_name}],
                        ids=[doc_id]
                    )
                    st.success(f"Successfully stored in {domain} vault!")
                    st.balloons()

# --- SEARCH WORKSPACE ---
st.header("🔎 Knowledge Retrieval (RAG)")
search_tab, chat_tab = st.tabs(["Semantic Search", "AI Chat (Reasoning)"])

with search_tab:
    search_domain = st.selectbox("Search Domain", ["banking", "healthcare", "insurance"], key="search_domain")
    query = st.text_input("Ask a question about your documents (e.g., 'What is the patient diagnosis?')")
    
    if st.button("Search Knowledge Base"):
        if 'vdb' in st.session_state:
            from reasoning_engine import ReasoningEngine
            reasoner = ReasoningEngine(st.session_state['vdb'])
            
            with st.spinner("Thinking..."):
                response = reasoner.search_and_summarize(search_domain, query)
                st.markdown(response)
        else:
            st.error("Vector DB not initialized.")

with chat_tab:
    st.info("Full RAG Chat integration using GPT-4o coming in the next layer!")

# --- FOOTER ---
st.markdown("---")
st.caption("Built for Production | Enterprise RAG Pipeline v2.0")
