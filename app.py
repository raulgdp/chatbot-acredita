import os
import streamlit as st
from openai import OpenAI
import zipfile
import numpy as np
import pickle

# Inicialización segura
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

# Cargar base de conocimiento (embeddings precomputados con bge-small)
@st.cache_resource
def load_knowledge_base():
    try:
        # Descomprimir si es necesario
        if not os.path.exists("embeddings_db") and os.path.exists("embeddings_db.zip"):
            import zipfile
            with zipfile.ZipFile("embeddings_db.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        
        # Cargar datos
        embeddings = np.load("embeddings_db/embeddings.npy")
        with open("embeddings_db/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open("embeddings_db/sources.pkl", "rb") as f:
            sources = pickle.load(f)
        
        st.sidebar.success(f"✅ Base de conocimiento: {len(chunks)} chunks")
        return embeddings, chunks, sources
    except Exception as e:
        st.sidebar.warning("⚠️ Sin base de conocimiento (embeddings_db.zip no encontrado)")
        return None, None, None

embeddings, chunks, sources = load_knowledge_base()

# Búsqueda semántica (numpy puro - sin sentence-transformers en runtime)
def semantic_search(query, top_k=3):
    if embeddings is None or chunks is None:
        return [], []
    
    try:
        # Usar modelo ligero para consulta (se descarga una vez)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        
        # Similaridad coseno con numpy
        sims = np.dot(embeddings, query_emb)
        idx = np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in idx], [sources[i] for i in idx]
    except:
        return [], []

# Configuración API
IS_CLOUD = os.getenv("HOME") == "/home/appuser"
api_key = st.secrets["OPENAI_API_KEY"] if IS_CLOUD else os.getenv("OPENAI_API_KEY", "demo-key")
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
MODEL = "deepseek/deepseek-chat"  # ✅ Modelo válido

# Interfaz
st.title("🤖 ChatAcredita con Base de Conocimiento")
uploaded = st.file_uploader("📄 Sube PDF adicional", type=["pdf"])

if uploaded:
    import fitz
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    st.session_state.document_text = "".join(page.get_text() for page in doc)[:3000]
    doc.close()
    st.success("✅ PDF cargado")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando en documentos oficiales...")
        
        # ✅ RAG: Recuperar de base de conocimiento
        results, sources_found = semantic_search(prompt, top_k=2)
        context = "\n\n".join(results) if results else ""
        
        # Combinar con documento del usuario
        if st.session_state.document_text:
            context = f"Documento usuario:\n{st.session_state.document_text}\n\n{context}" if context else f"Documento usuario:\n{st.session_state.document_text}"
        
        if not context.strip():
            context = "No hay documentos disponibles."
        
        # Generar respuesta
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con base en el contexto."},
                    {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {prompt}"}
                ],
                max_tokens=400,
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
            placeholder.error(f"❌ Error: {str(e)[:100]}")