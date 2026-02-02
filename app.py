# app.py - Versión robusta con detección de dependencias faltantes
import os
import sys

# ════════════════════════════════════════════════════════════════════════════
# DETECCIÓN TEMPRANA DE DEPENDENCIAS FALTANTES (ANTES DE IMPORTAR OTRAS LIBRERÍAS)
# ════════════════════════════════════════════════════════════════════════════
required_packages = ["streamlit", "openai", "pymupdf", "sentence_transformers", "qdrant_client", "rank_bm25", "numpy"]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    st_warning = "⚠️ **Dependencias faltantes:**\n"
    st_warning += "\n".join([f"- `{pkg}`" for pkg in missing_packages])
    st_warning += "\n\n**Solución:**\n1. Verifica que `requirements.txt` esté en la raíz del repositorio\n2. Confirma formato UTF-8 sin BOM\n3. Haz 'Redeploy' en Streamlit Cloud"
    
    # Mostrar error sin crash (usando solo st si está disponible)
    try:
        import streamlit as st
        st.error(st_warning)
        st.stop()
    except:
        print(st_warning, file=sys.stderr)
        sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# AHORA SÍ IMPORTAR LIBRERÍAS (ya sabemos que existen)
# ════════════════════════════════════════════════════════════════════════════
import streamlit as st
from openai import OpenAI
import zipfile
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN SEGURA DE SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# ════════════════════════════════════════════════════════════════════════════
# CARGAR VECTORSTORE QDRANT + BM25
# ════════════════════════════════════════════════════════════════════════════
def ensure_qdrant_db():
    db_dir = "qdrant_db"
    db_zip = "qdrant_db.zip"
    
    if not os.path.exists(db_dir) and os.path.exists(db_zip):
        with st.spinner("📦 Descomprimiendo base de conocimiento..."):
            try:
                with zipfile.ZipFile(db_zip, 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.sidebar.success("✅ Base de conocimiento cargada")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error descomprimiendo: {str(e)[:100]}")
                return False
    elif os.path.exists(db_dir):
        st.sidebar.success("✅ Base de conocimiento disponible")
        return True
    else:
        st.sidebar.info("ℹ️ Sin base de conocimiento pre-cargada")
        return False

DB_AVAILABLE = ensure_qdrant_db()

@st.cache_resource
def load_qdrant_client():
    db_dir = "qdrant_db"
    
    if not os.path.exists(db_dir):
        st.sidebar.warning("⚠️ Carpeta qdrant_db no encontrada")
        return None
    
    try:
        client = QdrantClient(path=db_dir)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if "acreditacion" not in collection_names:
            st.sidebar.warning("⚠️ Colección 'acreditacion' no encontrada")
            return None
        
        st.sidebar.success("✅ Qdrant cargado | Colección: acreditacion")
        return client
        
    except Exception as e:
        st.sidebar.error(f"❌ Error Qdrant: {str(e)[:80]}")
        return None

qdrant_client = load_qdrant_client()

@st.cache_resource
def load_bm25():
    bm25_path = "qdrant_db/bm25_data.pkl"
    
    if not os.path.exists(bm25_path):
        st.sidebar.warning("⚠️ bm25_data.pkl no encontrado")
        return None, None, None
    
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        
        chunks = data["chunks"]
        sources = data["sources"]
        
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        st.sidebar.success(f"✅ BM25 cargado ({len(chunks)} chunks)")
        return bm25, chunks, sources
        
    except Exception as e:
        st.sidebar.error(f"❌ Error BM25: {str(e)[:80]}")
        return None, None, None

bm25, bm25_chunks, bm25_sources = load_bm25()

@st.cache_resource
def load_embedding_model():
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        st.sidebar.success("✅ Embedding model: BAAI/bge-small-en-v1.5 (384d)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error embeddings: {str(e)[:80]}")
        return None

embedding_model = load_embedding_model()

def hybrid_search(query, top_k=5):
    results = []
    sources_list = []
    
    if bm25 is not None:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                results.append(bm25_chunks[idx])
                sources_list.append(bm25_sources[idx])
    
    if qdrant_client is not None and embedding_model is not None:
        try:
            query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
            qdrant_results = qdrant_client.search(
                collection_name="acreditacion",
                query_vector=query_embedding.tolist(),
                limit=top_k * 2,
                with_payload=True
            )
            
            for result in qdrant_results:
                results.append(result.payload["text"])
                sources_list.append(result.payload["source"])
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error Qdrant: {str(e)[:50]}")
    
    if not results:
        return [], []
    
    unique_results = []
    unique_sources = []
    seen = set()
    
    for res, src in zip(results, sources_list):
        key = res[:100]
        if key not in seen:
            seen.add(key)
            unique_results.append(res)
            unique_sources.append(src)
    
    return unique_results[:top_k], unique_sources[:top_k]

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE API
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
    api_base = "https://openrouter.ai/api/v1".strip()

try:
    client = OpenAI(api_key=api_key, base_url=api_base)
except Exception as e:
    st.error(f"❌ Error OpenAI: {str(e)[:100]}")
    st.stop()

MODEL = "deepseek/deepseek-chat"

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

col_logo1, col_title, col_logo2 = st.columns([1, 2, 1])

with col_logo1:
    st.markdown("### 🎓 EISC")

with col_title:
    st.markdown(
        "<h1 style='text-align:center;color:#c00000;margin:0;'>🤖 ChatAcredita</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center;color:#1a5276;margin:0 0 10px 0;'>"
        "Asistente de Acreditación - EISC Univalle</h3>",
        unsafe_allow_html=True
    )

with col_logo2:
    st.markdown("### 🏛️ Univalle")

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Sistema RAG Híbrido")
    if bm25 is not None:
        st.markdown("✅ BM25 (búsqueda lexical)")
    if qdrant_client is not None:
        st.markdown("✅ Qdrant (búsqueda semántica)")
    if embedding_model is not None:
        st.markdown("✅ Embeddings: BAAI/bge-small-en-v1.5 (384d)")
    st.markdown("---")
    st.markdown("**Modelo LLM:**")
    st.markdown(f"`{MODEL}`")

uploaded = st.file_uploader("📄 Sube PDF sobre acreditación (opcional)", type=["pdf"])

if uploaded:
    try:
        import fitz
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)[:5000]
        doc.close()
        st.session_state.document_text = text
        st.session_state.document_name = uploaded.name
        st.success(f"✅ PDF procesado: {st.session_state.document_name}")
    except Exception as e:
        st.error(f"❌ Error PDF: {str(e)[:100]}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando en documentos...")
        
        relevant_chunks, chunk_sources = hybrid_search(prompt, top_k=4)
        
        context_parts = []
        all_sources = set()
        
        if relevant_chunks:
            rag_context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
            context_parts.append(f"Documentos:\n{rag_context}")
            all_sources.update(chunk_sources)
        
        if st.session_state.document_text:
            context_parts.append(f"Tu documento:\n{st.session_state.document_text}")
            all_sources.add(st.session_state.document_name)
        
        full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No hay documentos."
        
        if all_sources:
            sources_text = " | ".join([s for s in all_sources if s != "Desconocido"])
            placeholder.markdown(f"📚 Fuentes: {sources_text}\n\nGenerando respuesta...")
        else:
            placeholder.markdown("Generando respuesta...")
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con base en el contexto."},
                    {"role": "user", "content": f"Contexto:\n{full_context}\n\nPregunta: {prompt}"}
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
            error_msg = f"❌ Error: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if not st.session_state.messages:
    st.info("ℹ️ Sube documentos a 'pdfs/' y ejecuta entrenamiento_qdrant_bm25.py para crear tu base de conocimiento.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>"
    "Desarrollado por GUIA - EISC Univalle • RAG: BM25 + Qdrant (bge-small)</div>",
    unsafe_allow_html=True
)