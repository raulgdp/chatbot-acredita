# app.py - ChatAcredita: RAG 100% consistente entre local y Cloud
import os
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np
import streamlit.components.v1 as components
import unicodedata  # ✅ Para normalización Unicode

# ════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN UNICODE (CRÍTICO PARA ESPAÑOL)
# ════════════════════════════════════════════════════════════════════════════
def normalize_text(text):
    """
    Normaliza texto para consistencia 100% entre entornos:
    1. Elimina acentos (NFD + filtrar combinaciones)
    2. Convierte a minúsculas
    3. Elimina espacios extra
    4. Reemplaza saltos de línea múltiples
    """
    # NFD: Descomponer caracteres (á → a + ´)
    text = unicodedata.normalize('NFD', text)
    # Eliminar marcas de combinación (acentos)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Minúsculas + limpieza
    text = text.lower().strip()
    text = ' '.join(text.split())  # Eliminar espacios múltiples
    text = text.replace('\n\n', '\n').replace('\n', ' ')  # Normalizar saltos
    return text

# ════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN SEGURA DE SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "meta-llama/llama-3.1-70b-instruct"

# ════════════════════════════════════════════════════════════════════════════
# MODELOS DISPONIBLES
# ════════════════════════════════════════════════════════════════════════════
AVAILABLE_MODELS = {
    "meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B",
    "qwen/qwen3-30b-a3b": "Qwen3 30B",
    "latam-gpt/Wayra-Perplexity-Estimator-55M": "Wayra (LATAM-GPT)",
    "qwen/qwen3-235b-a22b-thinking-2507": "Qwen3 235B"
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B (Gratis)"
}

# ════════════════════════════════════════════════════════════════════════════
# SCROLL INMEDIATO (optimizado)
# ════════════════════════════════════════════════════════════════════════════
def scroll_to_response():
    components.html(
        """
        <script>
        setTimeout(() => {
            const msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (msgs.length > 0) {
                msgs[msgs.length - 1].scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }, 50);
        </script>
        """,
        height=0,
        width=0,
    )

# ════════════════════════════════════════════════════════════════════════════
# CONEXIÓN A QDRANT CLOUD + BM25 DINÁMICO (100% consistente)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_qdrant_and_bm25():
    IS_CLOUD = os.getenv("HOME") == "/home/appuser"
    
    if IS_CLOUD:
        if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
            st.error("❌ Configura QDRANT_URL y QDRANT_API_KEY en Secrets")
            st.stop()
        url = st.secrets["QDRANT_URL"].strip()
        api_key = st.secrets["QDRANT_API_KEY"].strip()
    else:
        url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        api_key = os.getenv("QDRANT_API_KEY", None)
        if api_key:
            api_key = api_key.strip()
    
    try:
        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections()
        if "acreditacion" not in [c.name for c in collections.collections]:
            st.error("❌ Colección 'acreditacion' no encontrada")
            st.stop()
        
        # ✅ CARGA OPTIMIZADA CON NORMALIZACIÓN
        all_points = client.scroll(
            collection_name="acreditacion",
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # ✅ NORMALIZACIÓN UNICODE DE TODOS LOS CHUNKS (CRÍTICO)
        chunks = [normalize_text(p.payload["text"]) for p in all_points]
        sources = [p.payload.get("source", "Documento") for p in all_points]
        
        # ✅ BM25 CON TOKENIZACIÓN CONSISTENTE
        tokenized_chunks = [chunk.split() for chunk in chunks]  # ✅ Sin .lower() (ya normalizado)
        bm25 = BM25Okapi(tokenized_chunks)
        
        st.sidebar.success(f"✅ {len(chunks)} chunks cargados (normalizados)")
        return client, bm25, chunks, sources
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
        st.stop()

qdrant_client, bm25, bm25_chunks, bm25_sources = load_qdrant_and_bm25()

# ════════════════════════════════════════════════════════════════════════════
# MODELO DE EMBEDDINGS (cacheado con normalización)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        st.sidebar.success("✅ BGE-BASE-EN-V1.5 (768d) cargado")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error modelo: {str(e)[:100]}")
        st.stop()

embedding_model = load_embedding_model()

# ════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA HÍBRIDA 100% CONSISTENTE
# ════════════════════════════════════════════════════════════════════════════
def hybrid_search(query, top_k=3):
    """
    Búsqueda híbrida con normalización Unicode para consistencia 100%
    """
    # ✅ NORMALIZAR QUERY ANTES DE CUALQUIER OPERACIÓN
    query_normalized = normalize_text(query)
    
    results = []
    sources_list = []
    
    # 1. BM25 con texto normalizado
    if bm25 is not None:
        tokenized_query = query_normalized.split()  # ✅ Ya normalizado
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                results.append(bm25_chunks[idx])
                sources_list.append(bm25_sources[idx])
    
    # 2. Qdrant semántico (solo si necesario)
    if qdrant_client is not None and embedding_model is not None and len(results) < top_k * 2:
        try:
            # ✅ Usar query ORIGINAL para embeddings (no normalizado)
            # Los embeddings deben preservar significado semántico
            query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
            qdrant_results = qdrant_client.query_points(
                collection_name="acreditacion",
                query=query_embedding.tolist(),
                limit=top_k * 2 - len(results),
                with_payload=True
            ).points
            
            for result in qdrant_results:
                # ✅ Normalizar el texto recuperado para consistencia
                text_normalized = normalize_text(result.payload["text"])
                results.append(text_normalized)
                sources_list.append(result.payload.get("source", "Documento"))
        except:
            pass
    
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
# CONFIGURACIÓN API
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ Configura OPENAI_API_KEY en Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"].strip()
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "demo-key").strip()
    api_base = "https://openrouter.ai/api/v1".strip()

try:
    client = OpenAI(api_key=api_key, base_url=api_base)
except Exception as e:
    st.error(f"❌ Error API: {str(e)[:150]}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if os.path.exists("data/univalle_logo.png"):
        st.image("data/univalle_logo.png", width=80)
with col2:
    st.markdown("<h1 style='text-align:center;color:#c00000;'>🤖 ChatAcredita</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#1a5276;'>EISC Univalle</h3>", unsafe_allow_html=True)
with col3:
    if os.path.exists("data/logo2.png"):
        st.image("data/logo2.png", width=100)

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

# Selector de modelo (simplificado)
with st.sidebar:
    st.markdown("### 🤖 Modelo LLM")
    model_key = st.selectbox(
        "Modelo:",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed"
    )
    st.session_state.selected_model = model_key
    st.markdown(f"**Actual:** {AVAILABLE_MODELS[model_key]}")

uploaded = st.file_uploader("📄 Sube PDF sobre acreditación", type=["pdf"])
if uploaded:
    try:
        import fitz
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        # ✅ NORMALIZAR TEXTO DEL PDF SUBIDO
        text = normalize_text("".join(page.get_text() for page in doc)[:5000])
        doc.close()
        st.session_state.document_text = text
        st.session_state.document_name = uploaded.name
        st.success(f"✅ PDF procesado: {uploaded.name}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ✅ SCROLL INMEDIATO AL ENVIAR
    scroll_to_response()
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando en documentos...")
        
        # ✅ BÚSQUEDA CON NORMALIZACIÓN
        relevant_chunks, sources = hybrid_search(prompt, top_k=3)
        
        # Construir contexto
        context_parts = []
        if relevant_chunks:
            context_parts.append("Documentos oficiales:\n" + "\n\n".join(relevant_chunks))
        if st.session_state.document_text:
            context_parts.append(f"Tu documento:\n{st.session_state.document_text}")
        
        full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No hay contexto."
        
        # Mostrar fuentes
        if sources:
            placeholder.markdown(f"📚 Fuentes: {' | '.join(set(sources))}\n\nGenerando respuesta...")
        else:
            placeholder.markdown("Generando respuesta...")
        
        try:
            stream = client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con el contexto."},
                    {"role": "user", "content": f"Contexto:\n{full_context}\n\nPregunta: {prompt}"}
                ],
                max_tokens=500,
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
            scroll_to_response()
            
        except Exception as e:
            placeholder.error(f"❌ Error: {str(e)[:100]}")
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)[:100]}"})
            scroll_to_response()

if not st.session_state.messages:
    st.info("👋 ¡Hola! Soy **ChatAcredita**, tu asistente de acreditación de la **EISC**.\n\n✅ RAG 100% consistente (local + Cloud)\n✅ Normalización Unicode para español\n✅ Scroll inmediato al responder")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>"
    "Desarrollado por <strong>GUIA</strong> - EISC Univalle • RAG consistente 100%</div>",
    unsafe_allow_html=True
)