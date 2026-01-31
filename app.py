# app.py - RAG de alta calidad para Streamlit Cloud
import os
import streamlit as st
from openai import OpenAI
import fitz
import zipfile
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ════════════════════════════════════════════════════════════════════════════
# DESCOMPRIMIR CHROMA_DB AL INICIO
# ════════════════════════════════════════════════════════════════════════════
def ensure_chroma_db():
    chroma_dir = "chroma_db"
    chroma_zip = "chroma_db.zip"
    
    if not os.path.exists(chroma_dir) and os.path.exists(chroma_zip):
        with st.spinner("📦 Descomprimiendo base de conocimiento..."):
            try:
                with zipfile.ZipFile(chroma_zip, 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.sidebar.success("✅ Base de conocimiento cargada")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)[:80]}")
                return False
    elif os.path.exists(chroma_dir):
        st.sidebar.success("✅ Base de conocimiento disponible")
        return True
    else:
        st.sidebar.warning("⚠️ Sin base de conocimiento")
        return False

CHROMA_AVAILABLE = ensure_chroma_db()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN API
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ Configura OPENAI_API_KEY en Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    api_base = "https://openrouter.ai/api/v1"

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "deepseek/deepseek-v3.2"  # ✅ Usar DeepSeek como en tu sistema original

# ════════════════════════════════════════════════════════════════════════════
# CARGAR COMPONENTES RAG DE ALTA CALIDAD
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_rag_components():
    """Carga vectorstore + BM25 + reranker"""
    if not CHROMA_AVAILABLE:
        return None, None, None
    
    try:
        # ✅ Mismo modelo bge-m3 que usabas con FAISS
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        
        # ✅ Cargar BM25 (igual que sistema original)
        bm25_path = os.path.join("chroma_db", "bm25_retriever.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
        else:
            bm25_retriever = None
        
        # ✅ Cargar reranker ligero (BAAI/bge-reranker-base funciona en Cloud)
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        
        st.sidebar.info("🔍 RAG: bge-m3 + BM25 + Reranker")
        return vectorstore, bm25_retriever, reranker
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error cargando RAG: {str(e)[:80]}")
        return None, None, None

vectorstore, bm25_retriever, reranker = load_rag_components()

def hybrid_retrieve(query, top_k=5):
    """Recuperación híbrida (BM25 + vector) + reranking"""
    if not vectorstore:
        return [], "⚠️ RAG no disponible"
    
    try:
        # 1. Búsqueda BM25 (palabras clave)
        bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
        
        # 2. Búsqueda vectorial (semántica)
        vector_docs = vectorstore.similarity_search(query, k=10)
        
        # 3. Combinar y eliminar duplicados
        combined = {doc.page_content[:200]: doc for doc in bm25_docs + vector_docs}.values()
        combined = list(combined)[:10]
        
        if not combined:
            return [], "⚠️ No se encontró contexto relevante"
        
        # 4. ✅ Reranking con CrossEncoder (igual que sistema original)
        if reranker:
            pairs = [[query, doc.page_content] for doc in combined]
            scores = reranker.predict(pairs)
            scored = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in scored[:top_k]]
        else:
            reranked_docs = combined[:top_k]
        
        context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(reranked_docs)])
        sources = set([doc.metadata.get("source", "Desconocido") for doc in reranked_docs])
        
        return reranked_docs, context, sources
        
    except Exception as e:
        return [], f"⚠️ Error en recuperación: {str(e)[:100]}", set()

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ (mantén tu interfaz actual, solo modifica la sección de recuperación)
# ════════════════════════════════════════════════════════════════════════════
# ... [tu código de interfaz existente] ...

# EN LA SECCIÓN DE RESPUESTA, REEMPLAZA LA RECUPERACIÓN CON:
if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando información relevante...")
        
        # ✅ RECUPERACIÓN HÍBRIDA DE ALTA CALIDAD
        docs, base_context, sources = hybrid_retrieve(prompt, top_k=4)
        
        # Agregar documento subido por usuario
        extra_context = st.session_state.document_text if st.session_state.document_text else ""
        full_context = (base_context + "\n\n" + extra_context) if extra_context else base_context
        
        # Mostrar fuentes
        sources_text = ", ".join(sources) if sources else "Documento subido por usuario"
        placeholder.markdown(f"📚 Fuentes: {sources_text}\n\nGenerando respuesta...")
        
        # Generar respuesta con DeepSeek
        try:
            stream = client.chat.completions.create(
                model=MODEL,  # ✅ deepseek/deepseek-v3.2
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, experto en acreditación de la EISC. Responde con precisión basado SOLO en el contexto."},
                    {"role": "user", "content": f"Contexto:\n{full_context}\n\nPregunta: {prompt}\n\nRespuesta:"}
                ],
                max_tokens=800,
                temperature=0.2,  # ✅ Más bajo para respuestas precisas (igual que sistema original)
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
            error_msg = f"❌ Error: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})