import uuid
import os
import tempfile
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
from dotenv import load_dotenv

# Import your original RAG functions
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        border-radius: 20px;
        padding: 12px 20px;
        margin: 10px 0;
        color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    
    .ai-message {
        background: white;
        border-radius: 20px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #2c3e50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    
    /* PDF source message */
    .ai-message-pdf {
        border-left: 5px solid #27ae60;
    }
    
    /* AI general message */
    .ai-message-general {
        border-left: 5px solid #f39c12;
    }
    
    /* User name styling */
    .user-name {
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 5px;
        opacity: 0.9;
    }
    
    .ai-name {
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 8px;
        color: #666;
    }
    
    /* Badge styling */
    .pdf-badge {
        background: #27ae60;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-left: 10px;
    }
    
    .ai-badge {
        background: #f39c12;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-left: 10px;
    }
    
    /* Message text */
    .message-text {
        font-size: 15px;
        line-height: 1.5;
    }
    
    .ai-warning-text {
        background: #fff3cd;
        color: #856404;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    
    .feature-card h3 {
        color: #4A90E2;
        margin-bottom: 12px;
        font-size: 1.2rem;
    }
    
    .feature-card p {
        color: #555;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Status indicators */
    .status-success {
        background: #27ae60;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        font-size: 13px;
    }
    
    .status-warning {
        background: #f39c12;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        font-size: 13px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        font-size: 14px;
    }
    
    .stTextArea textarea:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
    }
    
    /* About page content */
    .about-content {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        color: #2c3e50;
    }
    
    .about-content h2 {
        color: #4A90E2;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .about-content p, .about-content li {
        color: #555;
        line-height: 1.6;
    }
    
    /* Features page content */
    .features-content {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px;
        color: #2c3e50;
    }
    
    .features-content h2 {
        color: #4A90E2;
        margin-bottom: 15px;
    }
    
    .features-content li {
        color: #555;
        line-height: 1.8;
    }
    
    .features-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .features-table td {
        padding: 8px;
        border-bottom: 1px solid #eee;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'collection_id' not in st.session_state:
    st.session_state.collection_id = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Initialize embeddings and model
@st.cache_resource
def initialize_components():
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    model = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.3
    )
    return embeddings, model

def process_pdf(file_path, embeddings):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(docs)
    collection_id = str(uuid.uuid4())
    
    PERSIST_DIR = "vector_db"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"{PERSIST_DIR}/{collection_id}",
        collection_name=collection_id
    )
    
    return vector_db, chunks, collection_id

def build_retriever(vector_db, chunks):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    
    vector_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )
    
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    return hybrid

# Initialize components
embeddings, model = initialize_components()

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional PDF Assistant.
Use Context and Chat History to answer.

Rules:      
    1. First search for the answer in the provided PDF context.
    2. If the answer is available in the PDF context, answer using the PDF only.
    3. If the answer is from the PDF, mention the source page number if available.
    4. If the answer is NOT available in the PDF context, then answer using your general AI knowledge.
    5. When answering from general AI knowledge, you MUST start your response with exactly this line:

    [AI Generated Answer - Not Found In PDF]

    Then continue with your answer.

    6. Never pretend that AI-generated information came from the PDF.
    7. Use chat history when helpful

Context:
{context}

Chat History:
{history}"""),
    ("human", "{question}")
])

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=70)
    st.title("📚 PDF RAG")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Menu",
        options=["💬 Chat", "ℹ️ About", "⭐ Features"],
        icons=["chat-dots", "info-circle", "star"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4A90E2", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "5px 0", "color": "white"},
            "nav-link-selected": {"background-color": "#4A90E2", "border-radius": "8px"},
        }
    )
    
    st.markdown("---")
    
    # File upload
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf", key="pdf_uploader")
    
    if uploaded_file is not None:
        if st.button("🔄 Process PDF", use_container_width=True, key="process_btn"):
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    vector_db, chunks, collection_id = process_pdf(tmp_path, embeddings)
                    retriever = build_retriever(vector_db, chunks)
                    
                    st.session_state.vector_db = vector_db
                    st.session_state.chunks = chunks
                    st.session_state.collection_id = collection_id
                    st.session_state.retriever = retriever
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    
                    st.success(f"✅ {uploaded_file.name}")
                    
                    os.unlink(tmp_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Status
    if st.session_state.pdf_processed:
        st.markdown(f'<div class="status-success">✅ {st.session_state.current_pdf_name[:30]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ No PDF loaded</div>', unsafe_allow_html=True)
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        st.rerun()

# Main content
if selected == "💬 Chat":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF RAG Assistant</h1>
        <p>Upload PDF and ask questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("💡 Ask a question about your PDF...")
        else:
            for message in st.session_state.chat_history:
                if message.startswith("Human:"):
                    user_text = message[6:]
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="user-name">👤 You</div>
                        <div class="message-text">{user_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    ai_text = message[4:]
                    
                    # Check if this is AI generated answer (contains the special tag)
                    if "[AI Generated Answer - Not Found In PDF]" in ai_text:
                        # Remove the tag for clean display
                        clean_text = ai_text.replace("[AI Generated Answer - Not Found In PDF]", "").strip()
                        st.markdown(f"""
                        <div class="ai-message ai-message-general">
                            <div class="ai-name">
                                🤖 AI Assistant 
                                <span class="ai-badge">General Knowledge</span>
                            </div>
                            <div class="ai-warning-text">⚠️ [AI Generated Answer - Not Found In PDF]</div>
                            <div class="message-text">{clean_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # This is PDF-sourced answer
                        st.markdown(f"""
                        <div class="ai-message ai-message-pdf">
                            <div class="ai-name">
                                📚 AI Assistant 
                                <span class="pdf-badge">From PDF</span>
                            </div>
                            <div class="message-text">{ai_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input area
    if st.session_state.pdf_processed:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            input_key = f"input_{st.session_state.input_key}"
            user_query = st.text_area("Ask a question:", 
                                     placeholder="Example: What is this document about?",
                                     height=80,
                                     key=input_key)
        
        with col2:
            st.write("")
            st.write("")
            send_button = st.button("📤 Send", use_container_width=True, type="primary")
        
        if send_button and user_query:
            with st.spinner("Searching PDF..."):
                try:
                    docs = st.session_state.retriever.invoke(user_query)
                    context_text = "\n\n".join(doc.page_content for doc in docs)
                    formatted_history = "\n".join(st.session_state.chat_history[-6:])
                    
                    chain = QA_PROMPT | model
                    response = chain.invoke({
                        "context": context_text,
                        "history": formatted_history,
                        "question": user_query
                    })
                    
                    answer = response.content
                    
                    st.session_state.chat_history.append(f"Human: {user_query}")
                    st.session_state.chat_history.append(f"AI: {answer}")
                    
                    # Increment key to clear input
                    st.session_state.input_key += 1
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("📚 **Please upload a PDF file in the sidebar to start!**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🔍 Hybrid Search</h3>
                <p>BM25 + Vector search for accurate results</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 Llama 3.3 70B</h3>
                <p>Powered by Groq's LLM</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>📚 RAG Architecture</h3>
                <p>Answers grounded in your PDF</p>
            </div>
            """, unsafe_allow_html=True)

elif selected == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>📖 About PDF RAG Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-content">
        <h2>🚀 What is this?</h2>
        <p>Intelligent PDF chat assistant using <strong>Retrieval-Augmented Generation (RAG)</strong> to answer questions about your PDF documents.</p>
        
        <h2>🔧 How it works</h2>
        <ol>
            <li><strong>PDF Processing</strong> - Splits PDF into chunks for retrieval</li>
            <li><strong>Hybrid Search</strong> - BM25 (keyword) + Vector (semantic) search</li>
            <li><strong>AI Generation</strong> - Llama 3.3 70B generates responses</li>
            <li><strong>Source Attribution</strong> - Clear indication of answer source</li>
        </ol>
        
        <h2>✨ Features</h2>
        <ul>
            <li>🔍 Hybrid Retrieval (BM25 + Vector)</li>
            <li>💬 Conversational AI with context memory</li>
            <li>📊 Clear source attribution</li>
            <li>📄 Page number references</li>
            <li>🎨 Modern responsive UI</li>
        </ul>
        
        <h2>🛠️ Technologies</h2>
        <ul>
            <li>Streamlit - Web interface</li>
            <li>LangChain - RAG framework</li>
            <li>ChromaDB - Vector database</li>
            <li>Groq - Llama 3.3 70B inference</li>
            <li>HuggingFace - Embeddings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif selected == "⭐ Features":
    st.markdown("""
    <div class="main-header">
        <h1>⭐ Features & Specifications</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="features-content">
            <h2>🎯 Core Features</h2>
            <ul>
                <li>📄 PDF Processing - Any PDF document</li>
                <li>🔍 Hybrid Search - BM25 + Vector retrieval</li>
                <li>💬 Conversational AI - Chat history maintained</li>
                <li>📊 Source Attribution - Clear answer sources</li>
                <li>⚡ Fast Responses - Optimized pipeline</li>
                <li>🎨 Modern UI - Beautiful design</li>
            </ul>
            
            <h2>🤖 AI Capabilities</h2>
            <ul>
                <li>Context Understanding</li>
                <li>Multi-turn Dialog</li>
                <li>Fallback Knowledge (AI)</li>
                <li>Page Number References</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="features-content">
            <h2>📊 Technical Specs</h2>
            <table class="features-table">
                <tr><td><strong>Chunk Size</strong></td><td>1000 characters</td></tr>
                <tr><td><strong>Overlap</strong></td><td>200 characters</td></tr>
                <tr><td><strong>Vector Search</strong></td><td>MMR (k=5)</td></tr>
                <tr><td><strong>BM25 Results</strong></td><td>10 documents</td></tr>
                <tr><td><strong>Weights</strong></td><td>0.4 BM25 / 0.6 Vector</td></tr>
                <tr><td><strong>Model</strong></td><td>Llama 3.3 70B</td></tr>
                <tr><td><strong>Temperature</strong></td><td>0.3</td></tr>
            </table>
            
            <h2>🎨 UI Features</h2>
            <ul>
                <li>Responsive Design</li>
                <li>Animated Messages</li>
                <li>Status Indicators</li>
                <li>Easy Navigation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 15px; font-size: 12px;">
    PDF RAG Assistant | Created with ❤️ by Ashish Pal | Powered by Streamlit & LangChain
</div>
""", unsafe_allow_html=True)