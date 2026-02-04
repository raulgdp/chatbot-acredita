# app.py - ChatAcredita: BM25 dinámico + Qdrant Cloud (768d) + Llama 3.1 70B + Scroll automático
import os
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np
import streamlit.components.v1 as components

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
# SCROLL AUTOMÁTICO AL FINAL (después de cada respuesta)
# ════════════════════════════════════════════════════════════════════════════
def scroll_to_bottom():
    """Fuerza el scroll al final de la página usando JavaScript"""
    components.html(
        """
        <script>
            setTimeout(function() {
                const mainBlock = window.parent.document.querySelector('section.main');
                if (mainBlock) {
                    mainBlock.scrollTop = mainBlock.scrollHeight;
                }
                window.parent.scrollTo(0, document.body.scrollHeight);
            }, 100);
        </script>
        """,
        height=0,
        width=0,
    )

# ════════════════════════════════════════════════════════════════════════════
# CONEXIÓN A QDRANT CLOUD + BM25 DINÁMICO (sin archivos .pkl locales)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_qdrant_and_bm25():
    """
    Conexión a Qdrant Cloud + extracción de chunks para BM25 dinámico
    ✅ Sin dependencia de archivos .pkl locales
    ✅ BM25 construido en memoria al inicio
    """
    IS_CLOUD = os.getenv("HOME") == "/home/appuser"
    
    # Configurar credenciales
    if IS_CLOUD:
        if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
            st.error("❌ Configura QDRANT_URL y QDRANT_API_KEY en Settings → Secrets")
            st.stop()
        url = st.secrets["QDRANT_URL"].strip()
        api_key = st.secrets["QDRANT_API_KEY"].strip()
    else:
        url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        api_key = os.getenv("QDRANT_API_KEY", None)
        if api_key:
            api_key = api_key.strip()
    
    try:
        # Conectar a Qdrant Cloud
        client = QdrantClient(url=url, api_key=api_key)
        
        # Verificar colección
        collections = client.get_collections()
        if "acreditacion" not in [c.name for c in collections.collections]:
            st.error("❌ Colección 'acreditacion' no encontrada en Qdrant Cloud")
            st.stop()
        
        st.sidebar.success("✅ Qdrant Cloud conectado (768d)")
        
        # ✅ EXTRAER TODOS LOS CHUNKS PARA BM25 DINÁMICO
        st.sidebar.info("🔄 Extrayendo chunks para BM25 dinámico...")
        all_points = client.scroll(
            collection_name="acreditacion",
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Extraer textos y fuentes
        chunks = [p.payload["text"] for p in all_points]
        sources = [p.payload.get("source", "Documento") for p in all_points]
        
        st.sidebar.success(f"✅ Cargados {len(chunks)} chunks para BM25")
        
        # ✅ CONSTRUIR BM25 EN MEMORIA (sin archivos .pkl)
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        st.sidebar.success("✅ BM25 dinámico construido en memoria")
        
        return client, bm25, chunks, sources
        
    except Exception as e:
        st.sidebar.error(f"❌ Error Qdrant Cloud: {str(e)[:120]}")
        if "403" in str(e) or "forbidden" in str(e).lower():
            st.sidebar.error("🔑 Verifica QDRANT_API_KEY en Secrets (sin espacios)")
        elif "404" in str(e):
            st.sidebar.error("🔗 Verifica QDRANT_URL en Secrets (sin espacios al final)")
        st.stop()

qdrant_client, bm25, bm25_chunks, bm25_sources = load_qdrant_and_bm25()

# ════════════════════════════════════════════════════════════════════════════
# MODELO DE EMBEDDINGS (BGE-BASE-EN-V1.5 - 768d)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embedding_model():
    """Carga BGE-BASE-EN-V1.5 (768 dimensiones)"""
    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        st.sidebar.success("✅ Embeddings: BAAI/bge-base-en-v1.5 (768d)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar modelo: {str(e)[:100]}")
        st.stop()

embedding_model = load_embedding_model()

# ════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA HÍBRIDA: BM25 DINÁMICO + QDRANT SEMÁNTICO
# ════════════════════════════════════════════════════════════════════════════
def hybrid_search(query, top_k=4):
    """
    RAG híbrido SIN ARCHIVOS .PKL:
    1. BM25: búsqueda lexical (construido dinámicamente en memoria)
    2. Qdrant: búsqueda semántica (768d)
    """
    results = []
    sources_list = []
    
    # 1. Búsqueda BM25 lexical (dinámico)
    if bm25 is not None:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                results.append(bm25_chunks[idx])
                sources_list.append(bm25_sources[idx])
    
    # 2. Búsqueda Qdrant semántica (768d)
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
                sources_list.append(result.payload.get("source", "Documento"))
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
# CONFIGURACIÓN DE API - LLAMA 3.1 70B (MODELO VÁLIDO)
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
MODEL= "deepseek/deepseek-r1"


#MODEL = "meta-llama/llama-3.1-70b-instruct"  # ✅ Único modelo Llama 3.1 válido

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO CON LOGOS INSTITUCIONALES
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
    st.markdown("✅ BM25 dinámico (búsqueda lexical en memoria)")
    st.markdown("✅ Qdrant Cloud (búsqueda semántica 768d)")
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
        
        # ✅ RAG HÍBRIDO SIN ARCHIVOS .PKL
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
        
        full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No hay documentos disponibles."
        
        # ✅ F-STRINGS CORREGIDOS (mostrar modelo real, no literal {MODEL})
        if all_sources:
            sources_text = " | ".join([s for s in all_sources if s != "Desconocido"])
            placeholder.markdown(f"📚 Fuentes: {sources_text}\n\n🧠 Generando respuesta con **{MODEL}**...")
        else:
            placeholder.markdown(f"🧠 Generando respuesta con **{MODEL}**...")
        
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
            
            # ✅ SCROLL AUTOMÁTICO AL FINAL DESPUÉS DE LA RESPUESTA
            scroll_to_bottom()
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            scroll_to_bottom()  # Scroll también en errores

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **EISC**.
        
        ### 🚀 Sistema RAG Híbrido (sin archivos .pkl problemáticos):
        - **BM25 dinámico**: Búsqueda lexical construida en memoria desde Qdrant Cloud
        - **Qdrant Cloud**: Búsqueda semántica con embeddings BAAI/bge-base-en-v1.5 (768d)
        - **Deepseek-R1**: Respuestas de alta calidad y precisión
        
        ### 💡 Ventajas:
        - ✅ Sin dependencia de archivos locales (.pkl, .npy)
        - ✅ Scroll automático al final después de cada respuesta
        - ✅ Sin errores de rutas/archivos faltantes en Streamlit Cloud
        - ✅ BM25 + búsqueda semántica para máxima cobertura
        
        *Sube documentos adicionales para complementar la información oficial.*
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;padding:10px 0;'>"
    "Desarrollado por <strong>GUIA</strong> - Grupo de Univalle en Inteligencia Artificial | "
    "EISC Univalle • RAG: BM25 dinámico + Qdrant Cloud (bge-base 768d)</div>",
    unsafe_allow_html=True
)