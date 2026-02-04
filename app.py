# app.py - ChatAcredita con RAG Híbrido: BM25 + Qdrant Cloud (bge-base 768d) + Llama 3.1 70B
import os
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
# CARGAR BM25 (ROBUSTO: maneja ambas estructuras de carpetas)
# ════════════════════════════════════════════════════════════════════════════
def ensure_embeddings_db():
    """Descomprime embeddings_db.zip y maneja ambas estructuras posibles"""
    db_zip = "embeddings_db.zip"
    db_dir = "embeddings_db"
    
    if not os.path.exists(db_dir) and os.path.exists(db_zip):
        with st.spinner("📦 Descomprimiendo base de conocimiento..."):
            try:
                with zipfile.ZipFile(db_zip, 'r') as zip_ref:
                    zip_ref.extractall(db_dir)
                st.sidebar.success("✅ Base de conocimiento descomprimida")
                
                # ✅ Detectar y corregir estructura con carpeta intermedia
                possible_subdirs = ["embeddings_export", "embeddings_db"]
                for subdir in possible_subdirs:
                    subdir_path = os.path.join(db_dir, subdir)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        # Mover archivos a raíz de embeddings_db/
                        import shutil
                        for item in os.listdir(subdir_path):
                            s = os.path.join(subdir_path, item)
                            d = os.path.join(db_dir, item)
                            if os.path.isdir(s):
                                shutil.copytree(s, d, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s, d)
                        # Eliminar carpeta intermedia
                        shutil.rmtree(subdir_path)
                        st.sidebar.info(f"🔄 Estructura corregida: eliminada carpeta '{subdir}'")
                        break
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error descomprimiendo: {str(e)[:100]}")
                return False
    elif os.path.exists(db_dir):
        st.sidebar.success("✅ Base de conocimiento disponible")
        return True
    else:
        st.sidebar.warning("⚠️ embeddings_db.zip no encontrado")
        return False

DB_AVAILABLE = ensure_embeddings_db()

@st.cache_resource
def load_bm25():
    """Carga BM25 desde embeddings_db (maneja múltiples rutas posibles)"""
    # Buscar bm25_data.pkl en rutas posibles
    possible_paths = [
        "embeddings_db/bm25_data.pkl",
        "embeddings_db/embeddings_export/bm25_data.pkl",
        "embeddings_db/data/bm25_data.pkl"
    ]
    
    bm25_path = None
    for path in possible_paths:
        if os.path.exists(path):
            bm25_path = path
            break
    
    if not bm25_path:
        st.sidebar.error("❌ bm25_data.pkl no encontrado en ninguna ruta")
        st.sidebar.info("💡 Ejecuta: Compress-Archive -Path 'embeddings_export\\*' -DestinationPath 'embeddings_db.zip'")
        return None, None, None
    
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        
        chunks = data["chunks"]
        sources = data["sources"]
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        st.sidebar.success(f"✅ BM25 cargado ({len(chunks)} chunks | 768d)")
        return bm25, chunks, sources
        
    except Exception as e:
        st.sidebar.error(f"❌ Error BM25: {str(e)[:100]}")
        return None, None, None

bm25, bm25_chunks, bm25_sources = load_bm25()

@st.cache_resource
def load_qdrant_cloud_client():
    """Conexión SEGURA a Qdrant Cloud"""
    IS_CLOUD = os.getenv("HOME") == "/home/appuser"
    
    if IS_CLOUD:
        if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
            st.sidebar.error("❌ Configura QDRANT_URL y QDRANT_API_KEY en Secrets")
            return None
        url = st.secrets["QDRANT_URL"].strip()  # ✅ Eliminar espacios
        api_key = st.secrets["QDRANT_API_KEY"].strip()  # ✅ Eliminar espacios
    else:
        url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        api_key = os.getenv("QDRANT_API_KEY", None)
        if api_key:
            api_key = api_key.strip()
    
    try:
        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "acreditacion" not in collection_names:
            st.sidebar.warning("⚠️ Colección 'acreditacion' no encontrada")
            return None
        
        st.sidebar.success("✅ Qdrant Cloud: conexión exitosa (768d)")
        return client
        
    except Exception as e:
        st.sidebar.error(f"❌ Qdrant Cloud error: {str(e)[:100]}")
        if "403" in str(e) or "forbidden" in str(e).lower():
            st.sidebar.error("🔑 Verifica QDRANT_API_KEY en Secrets")
        elif "404" in str(e):
            st.sidebar.error("🔗 Verifica QDRANT_URL en Secrets (sin espacios al final)")
        return None

qdrant_client = load_qdrant_cloud_client()

@st.cache_resource
def load_embedding_model():
    """✅ BAAI/bge-base-en-v1.5 (768d) - modelo de alta calidad para inglés/español técnico"""
    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        st.sidebar.success("✅ Embedding model: BAAI/bge-base-en-v1.5 (768d)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error modelo: {str(e)[:100]}")
        return None

embedding_model = load_embedding_model()

def hybrid_search(query, top_k=4):
    """RAG híbrido: BM25 (local) + Qdrant Cloud (768d)"""
    results = []
    sources_list = []
    
    # 1. Búsqueda BM25 (lexical)
    if bm25 is not None:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                results.append(bm25_chunks[idx])
                sources_list.append(bm25_sources[idx])
    
    # 2. Búsqueda Qdrant Cloud (semántica 768d)
    if qdrant_client is not None and embedding_model is not None:
        try:
            query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
            qdrant_results = qdrant_client.query_points(
                collection_name="acreditacion",
                query=query_embedding.tolist(),
                limit=top_k * 2,
                with_payload=True
            ).points
            
            for result in qdrant_results:
                results.append(result.payload["text"])
                sources_list.append(result.payload["source"])
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error Qdrant: {str(e)[:50]}")
    
    if not results:
        return [], []
    
    # Eliminar duplicados
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
# CONFIGURACIÓN DE API - MODELOS VÁLIDOS
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ Configura OPENAI_API_KEY en Settings → Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"].strip()
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "demo-key").strip()
    api_base = "https://openrouter.ai/api/v1".strip()

try:
    client = OpenAI(api_key=api_key, base_url=api_base)
except Exception as e:
    st.error(f"❌ Error OpenAI: {str(e)[:150]}")
    st.stop()

# ✅ MODELO VÁLIDO (llama-4-scout NO EXISTE)
#MODEL = "meta-llama/llama-3.1-70b-instruct"  # ✅ Único modelo Llama 3.1 válido en OpenRouter
MODEL = "deepseek/deepseek-r1"

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO CON LOGOS
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

col_logo1, col_title, col_logo2 = st.columns([1, 2, 1])

with col_logo1:
    if os.path.exists("data/univalle_logo.png"):
        st.image("data/univalle_logo.png", width=80)
    else:
        st.markdown("### 🎓 Univalle")

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
    if os.path.exists("data/logo2.png"):
        st.image("data/logo2.png", width=100)
    else:
        st.markdown("### 🤖 GUIA")

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Sistema RAG Híbrido")
    st.markdown("✅ BM25 (búsqueda lexical local)")
    st.markdown("✅ Qdrant Cloud (semántica 768d)")
    st.markdown("✅ Embeddings: BAAI/bge-base-en-v1.5")
    st.markdown("---")
    st.markdown(f"**Modelo LLM:** `{MODEL}`")

uploaded = st.file_uploader("📄 Sube PDF adicional sobre acreditación", type=["pdf"])

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

if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando en documentos oficiales...")
        
        # ✅ RAG HÍBRIDO: BM25 local + Qdrant Cloud (768d)
        relevant_chunks, chunk_sources = hybrid_search(prompt, top_k=4)
        
        context_parts = []
        all_sources = set()
        
        if relevant_chunks:
            rag_context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
            context_parts.append(f"Documentos oficiales:\n{rag_context}")
            all_sources.update(chunk_sources)
        
        if st.session_state.document_text:
            context_parts.append(f"Tu documento:\n{st.session_state.document_text}")
            all_sources.add(st.session_state.document_name)
        
        full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No hay documentos."
        
        if all_sources:
            sources_text = " | ".join([s for s in all_sources if s != "Desconocido"])
            placeholder.markdown(f"📚 Fuentes: {sources_text}\n\nGenerando respuesta con `{MODEL}`")
        else:
            # ✅ CORREGIDO: f-string para interpolar la variable MODEL
            placeholder.markdown(f"Generando respuesta con `{MODEL}`")
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres ChatAcredita, asistente especializado en acreditación de programas de la "
                            "Escuela de Ingeniería de Sistemas y Computación de la Universidad del Valle. "
                            "Responde SOLO con base en el contexto proporcionado. Sé preciso y profesional."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Contexto:\n{full_context}\n\nPregunta: {prompt}"
                    }
                ],
                max_tokens=600,
                temperature=0.2,
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
            if "404" in str(e) and "model" in str(e).lower():
                st.error("""
                🔑 **ERROR DE MODELO:**
                'meta-llama/llama-4-scout' NO EXISTE en OpenRouter.
                
                ✅ Usa SOLO estos modelos válidos:
                • meta-llama/llama-3.1-70b-instruct (recomendado)
                • meta-llama/llama-3.2-3b-instruct:free (gratuito)
                
                Lista completa: https://openrouter.ai/models
                """)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **EISC**.
        
        ### 🚀 Sistema RAG Híbrido:
        - **BM25**: Búsqueda lexical por palabras clave (local)
        - **Qdrant Cloud**: Búsqueda semántica con embeddings BAAI/bge-base-en-v1.5 (768d)
        - **Llama 3.1 70B**: Respuestas de alta calidad y precisión
        
        ### 💡 Solución de problemas comunes:
        - ⚠️ **"bm25_data.pkl no encontrado"** → Ejecuta en PowerShell:
          ```powershell
          Compress-Archive -Path "embeddings_export\\*" -DestinationPath "embeddings_db.zip" -Force
          ```
        - 🔑 **Error Qdrant Cloud** → Verifica Secrets sin espacios al final de URLs
        
        
        *Sube documentos adicionales para complementar la información oficial.*
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;padding:10px 0;'>"
    "Desarrollado por <strong>GUIA</strong> - Grupo de Univalle en Inteligencia Artificial | "
    "EISC Univalle • RAG: BM25 + Qdrant Cloud (bge-base 768d) + Llama 3.1 70B</div>",
    unsafe_allow_html=True
)