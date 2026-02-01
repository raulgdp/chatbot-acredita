# app.py - ChatAcredita con RAG funcional usando embeddings manuales (sin ChromaDB/LangChain)
import os
import streamlit as st
from openai import OpenAI
import zipfile
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN SEGURA DE SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ════════════════════════════════════════════════════════════════════════════
# CARGAR BASE DE DATOS VECTORIAL (EMBEDDINGS MANUALES)
# ════════════════════════════════════════════════════════════════════════════
def ensure_embeddings_db():
    """Descomprime embeddings_db.zip si no existe la carpeta"""
    db_dir = "embeddings_db"
    db_zip = "embeddings_db.zip"
    
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

DB_AVAILABLE = ensure_embeddings_db()

@st.cache_resource
def load_vector_db():
    """Carga embeddings, chunks y fuentes desde disco"""
    db_dir = "embeddings_db"
    
    if not os.path.exists(db_dir):
        return None, None, None
    
    try:
        # Cargar embeddings (numpy array)
        embeddings_path = os.path.join(db_dir, "embeddings.npy")
        if not os.path.exists(embeddings_path):
            st.sidebar.warning("⚠️ embeddings.npy no encontrado")
            return None, None, None
        
        embeddings = np.load(embeddings_path)
        
        # Cargar chunks (texto)
        chunks_path = os.path.join(db_dir, "chunks.pkl")
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        
        # Cargar fuentes (nombres de PDFs)
        sources_path = os.path.join(db_dir, "sources.pkl")
        with open(sources_path, "rb") as f:
            sources = pickle.load(f)
        
        # Verificar consistencia
        if len(embeddings) != len(chunks) or len(chunks) != len(sources):
            st.sidebar.warning("⚠️ Inconsistencia en los datos")
            return None, None, None
        
        st.sidebar.info(f"📚 Base de conocimiento: {len(chunks)} chunks de {len(set(sources))} documentos")
        return embeddings, chunks, sources
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error cargando base de datos: {str(e)[:100]}")
        return None, None, None

# Cargar base de datos vectorial
embeddings, chunks, sources = load_vector_db()

@st.cache_resource
def load_embedding_model():
    """Carga el modelo de embeddings una sola vez"""
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return model
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error cargando modelo: {str(e)[:100]}")
        return None

embedding_model = load_embedding_model()

def semantic_search(query, top_k=3):
    """Búsqueda semántica usando similaridad coseno (sin dependencias externas)"""
    if embeddings is None or chunks is None or embedding_model is None:
        return [], []
    
    try:
        # Generar embedding de la consulta
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Calcular similaridad coseno
        similarities = np.dot(embeddings, query_embedding)
        
        # Obtener top_k índices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Recuperar chunks y fuentes
        results = [chunks[i] for i in top_indices]
        result_sources = [sources[i] for i in top_indices]
        
        return results, result_sources
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error en búsqueda: {str(e)[:100]}")
        return [], []

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
    api_base = "https://openrouter.ai/api/v1"

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "deepseek/deepseek-v3.2"  # ✅ Modelo válido y gratuito

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
    st.markdown("### 📚 Base de Conocimiento")
    if DB_AVAILABLE and embeddings is not None:
        st.markdown(f"✅ **{len(chunks)} chunks** indexados")
        st.markdown(f"📄 **{len(set(sources))} documentos** cargados")
        st.markdown("🔍 Búsqueda semántica activa")
    else:
        st.markdown("⚠️ Sin base de conocimiento")
        st.markdown("Sube `embeddings_db.zip` a GitHub")

uploaded = st.file_uploader("📄 Sube un PDF adicional sobre acreditación (opcional)", type=["pdf"])

if uploaded:
    try:
        import fitz
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        st.session_state.document_text = text[:5000]
        st.session_state.document_name = uploaded.name
        st.success(f"✅ PDF procesado: {st.session_state.document_name}")
    except Exception as e:
        st.error(f"❌ Error al procesar PDF: {str(e)[:100]}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando en base de conocimiento...")
        
        # ✅ RAG: Recuperar contexto de la base de datos vectorial
        relevant_chunks, chunk_sources = semantic_search(prompt, top_k=3)
        
        # Combinar contexto de base de datos + documento subido por usuario
        context_parts = []
        all_sources = set()
        
        if relevant_chunks:
            rag_context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
            context_parts.append(f"Documentos de referencia:\n{rag_context}")
            all_sources.update(chunk_sources)
        
        if st.session_state.document_text:
            context_parts.append(
                f"Documento adicional ({st.session_state.document_name}):\n"
                f"{st.session_state.document_text}"
            )
            all_sources.add(st.session_state.document_name)
        
        full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No hay documentos disponibles."
        
        # Mostrar fuentes
        if all_sources:
            sources_text = " | ".join([s for s in all_sources if s != "Desconocido"])
            placeholder.markdown(f"📚 Fuentes: {sources_text}\n\nGenerando respuesta...")
        else:
            placeholder.markdown("Generando respuesta...")
        
        # Generar respuesta con LLM
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres ChatAcredita, asistente especializado en acreditación de programas de la "
                            "Escuela de Ingeniería de Sistemas y Computación de la Universidad del Valle. "
                            "Responde SOLO con base en el contexto proporcionado. Sé preciso, conciso y profesional."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Contexto:\n{full_context}\n\nPregunta: {prompt}\n\nRespuesta:"
                    }
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

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **EISC**.
        
        ### 🚀 Cómo funciona:
        1. **Base de conocimiento pre-cargada** con documentos oficiales de acreditación
        2. **Búsqueda semántica** para encontrar información relevante
        3. **Respuestas precisas** basadas en documentos autorizados
        
        ### 💡 Ejemplos de preguntas:
        - "¿Cuáles son los requisitos para acreditar un programa de pregrado?"
        - "¿Qué estándares de calidad evalúa el CNA?"
        - "¿Cuál es el proceso de autoevaluación institucional?"
        
        *Puedes subir documentos adicionales para complementar la información.*
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;padding:10px 0;'>"
    "Desarrollado por <strong>GUIA</strong> - Grupo de Univalle en Inteligencia Artificial | "
    "EISC Univalle</div>",
    unsafe_allow_html=True
)