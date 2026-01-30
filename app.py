import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ ERROR: Configura OPENAI_API_KEY en Settings → Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    api_base = "https://openrouter.ai/api/v1"

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "deepseek/deepseek-v3.2"

# ════════════════════════════════════════════════════════════════════════════
# RAG CON CHROMADB (SIN FAISS)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_embedding_model():
    """Cargar modelo de embeddings una sola vez"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    """Crear cliente ChromaDB en memoria"""
    return chromadb.Client()

def create_vectorstore(text_chunks, collection_name="documents"):
    """Crear base de vectores a partir de chunks de texto"""
    client = get_chroma_client()
    embedding_model = get_embedding_model()
    
    # Eliminar colección si existe
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    # Crear nueva colección
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Generar embeddings y guardar
    embeddings = embedding_model.encode(text_chunks).tolist()
    ids = [f"id_{i}" for i in range(len(text_chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=text_chunks,
        ids=ids
    )
    
    return collection

def retrieve_similar(query, collection, top_k=3):
    """Recuperar documentos similares usando ChromaDB"""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results['documents'][0] if results['documents'] else []

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.markdown("### 🎓 EISC")
with col_title:
    st.markdown("<h1 style='text-align:center;color:#c00000;'>🤖 ChatAcredita</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#1a5276;'>Asistente de Acreditación - EISC Univalle</h3>", unsafe_allow_html=True)

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

# Estado de la app
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Subida y procesamiento de documento
uploaded = st.file_uploader("📄 Sube un PDF sobre acreditación", type=["pdf"])

if uploaded:
    try:
        with st.spinner("Leyendo y procesando PDF..."):
            # Extraer texto
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Dividir en chunks
            chunks = []
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size].strip()
                if len(chunk) > 100:  # Ignorar chunks muy pequeños
                    chunks.append(chunk)
            
            st.session_state.document_chunks = chunks
            
            # Crear vectorstore
            st.session_state.vectorstore = create_vectorstore(chunks, collection_name=f"doc_{hash(uploaded.name)}")
            
            st.success(f"✅ PDF procesado ({len(chunks)} fragmentos) - RAG activado")
    except Exception as e:
        st.error(f"❌ Error al procesar PDF: {str(e)[:150]}")

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando información relevante...")
        
        # Recuperación con RAG si hay vectorstore
        if st.session_state.vectorstore:
            relevant_docs = retrieve_similar(prompt, st.session_state.vectorstore, top_k=3)
            context = "\n\n".join(relevant_docs) if relevant_docs else "No se encontró información específica en el documento."
            retrieval_status = f"✅ Recuperados {len(relevant_docs)} fragmentos relevantes"
        else:
            context = "No hay documento cargado para consultar."
            retrieval_status = "⚠️ Sin documento cargado"
        
        placeholder.markdown(f"{retrieval_status}\n\nGenerando respuesta...")
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con base en el documento proporcionado. Si no hay información relevante, indícalo honestamente."},
                    {"role": "user", "content": f"Contexto del documento:\n{context}\n\nPregunta del usuario: {prompt}\n\nRespuesta:"}
                ],
                max_tokens=500,
                temperature=0.3,
                stream=True
            )
            
            answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"⚠️ Error de API: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>RAG con ChromaDB • Desarrollado por GUIA - EISC Univalle</div>", unsafe_allow_html=True)