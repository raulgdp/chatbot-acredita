# app.py - ChatAcredita con RAG funcional (100% compatible con Streamlit Cloud)
import os
import streamlit as st
from openai import OpenAI
import zipfile

# ════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN SEGURA DE SESSION STATE (PRIMERO QUE TODO)
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# ════════════════════════════════════════════════════════════════════════════
# CARGAR VECTORSTORE CHROMADB PARA RAG
# ════════════════════════════════════════════════════════════════════════════
def ensure_chroma_db():
    """Descomprime chroma_db.zip si chroma_db/ no existe"""
    chroma_dir = "chroma_db"
    chroma_zip = "chroma_db.zip"
    
    if not os.path.exists(chroma_dir) and os.path.exists(chroma_zip):
        with st.spinner(f"📦 Descomprimiendo base de conocimiento..."):
            try:
                with zipfile.ZipFile(chroma_zip, 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.sidebar.success("✅ Base de conocimiento cargada")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error descomprimiendo: {str(e)[:100]}")
                return False
    elif os.path.exists(chroma_dir):
        st.sidebar.success("✅ Base de conocimiento disponible")
        return True
    else:
        st.sidebar.info("ℹ️ Sin base de conocimiento pre-cargada")
        return False

# Descomprimir al inicio (antes de cualquier otra operación)
CHROMA_AVAILABLE = ensure_chroma_db()

@st.cache_resource
def load_vectorstore():
    """Carga el vectorstore ChromaDB si está disponible"""
    if not CHROMA_AVAILABLE:
        return None
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error cargando vectorstore: {str(e)[:100]}")
        return None

# Cargar vectorstore (cacheado)
vectorstore = load_vectorstore()

def retrieve_context(query, top_k=3):
    """Recupera contexto relevante del vectorstore + documento subido"""
    contexts = []
    sources = set()
    
    # 1. Recuperar de vectorstore ChromaDB (RAG)
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
            if docs:
                rag_context = "\n\n".join([
                    f"[{i+1}] {doc.page_content}" 
                    for i, doc in enumerate(docs)
                ])
                contexts.append(f"Documentos de referencia:\n{rag_context}")
                sources.update([
                    doc.metadata.get("source", "Desconocido") 
                    for doc in docs
                ])
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error en búsqueda semántica: {str(e)[:50]}")
    
    # 2. Agregar documento subido por usuario
    if st.session_state.document_text:
        contexts.append(
            f"Documento actual ({st.session_state.document_name}):\n"
            f"{st.session_state.document_text}"
        )
        sources.add(st.session_state.document_name)
    
    # Combinar contextos
    full_context = "\n\n---\n\n".join(contexts) if contexts else "No hay documentos disponibles."
    return full_context, sources

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
    api_base = "https://openrouter.ai/api#/v1"

client = OpenAI(api_key=api_key, base_url=api_base)

# ✅ MODELO VMODEL= "deepseek/deepseek-v3.2"ÁLIDO Y GRATUITO EN OPENROUTER (NO deepseek-v3.2)
MODEL = "mistralai/mistral-7b-instruct:free"  # ✅ 100% gratuito, sin costo

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

# Cabecera con texto institucional (logos opcionales)
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

# Panel lateral informativo
with st.sidebar:
    st.markdown("### 📚 Información")
    st.markdown("""
    **ChatAcredita** es un asistente especializado en procesos de acreditación de programas de la Escuela de Ingeniería de Sistemas y Computación.
    
    ### 📥 Cómo usar:
    1. Sube un documento PDF relacionado con acreditación
    2. Escribe tu pregunta en el chat
    3. Obtén respuestas basadas en documentos oficiales + tu PDF
    
    ### 💡 Consejo:
    Para mejores resultados, sube documentos oficiales como:
    - Guías de acreditación de la EISC
    - Resoluciones del CNA
    - Estándares de calidad institucionales
    """)
    
    if vectorstore:
        st.markdown("### ✅ RAG Activo")
        st.markdown("🔍 Búsqueda semántica disponible")
    else:
        st.markdown("### ⚠️ RAG No disponible")
        st.markdown("Sube chroma_db.zip a tu repositorio GitHub")

# Subida de documento
uploaded = st.file_uploader("📄 Sube un PDF sobre acreditación", type=["pdf"])

if uploaded:
    try:
        import fitz
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        st.session_state.document_text = text[:5000]
        st.session_state.document_name = uploaded.name
        st.success(f"✅ PDF procesado: {st.session_state.document_name}")
    except Exception as e:
        st.error(f"❌ Error al procesar PDF: {str(e)[:100]}")

# Mostrar historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Buscando información relevante...")
        
        # ✅ RECUPERACIÓN RAG (clave del sistema)
        full_context, sources = retrieve_context(prompt, top_k=3)
        
        # Mostrar fuentes utilizadas
        if sources:
            sources_text = " | ".join([s for s in sources if s != "Desconocido"])
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
                            "Responde SOLO con base en el contexto proporcionado. Sé preciso, conciso y profesional. "
                            "Si no hay información suficiente en el contexto, indícalo honestamente."
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
            
            # Mostrar respuesta en streaming
            answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer + "▌")
            
            # Mostrar respuesta final
            placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Diagnóstico específico para errores comunes
            error_str = str(e).lower()
            if "400" in error_str and "not a valid model" in error_str:
                st.error("""
                🔑 **Solución:** El modelo especificado no existe en OpenRouter.
                Usa 'mistralai/mistral-7b-instruct:free' (gratuito) o consulta:
                https://openrouter.ai/models
                """)
            elif "401" in error_str or "authentication" in error_str:
                st.error("""
                🔑 **Solución:** API key inválida o sin créditos.
                1. Regenera tu key en https://openrouter.ai/keys
                2. Configura Secrets en Streamlit Cloud
                """)

# Mensaje de bienvenida si no hay historial
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **Escuela de Ingeniería de Sistemas y Computación**.
        
        ### 🚀 Para empezar:
        1. **Sube un documento** relacionado con acreditación usando el botón de arriba
        2. **Haz tu pregunta** en el campo de chat
        3. **Obtén respuestas** basadas en documentos oficiales + tu documento
        
        ### 💡 Ejemplos de preguntas útiles:
        - "¿Cuáles son los requisitos para acreditar un programa de pregrado?"
        - "¿Qué estándares de calidad se evalúan en la acreditación?"
        - "¿Cuál es el proceso de autoevaluación institucional?"
        
        *Nota: Mis respuestas se basan en documentos oficiales de acreditación y el documento que me proporciones.*
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;padding:10px 0;'>"
    "Desarrollado por <strong>GUIA</strong> - Grupo de Univalle en Inteligencia Artificial | "
    "Escuela de Ingeniería de Sistemas y Computación | Universidad del Valle"
    "</div>",
    unsafe_allow_html=True
)