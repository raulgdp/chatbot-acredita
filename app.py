# app.py - ChatAcredita con RAG ChromaDB (funciona en Streamlit Cloud)
import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import zipfile
import shutil

# ════════════════════════════════════════════════════════════════════════════
# DESCOMPRIMIR CHROMA_DB AL INICIO (PARA STREAMLIT CLOUD)
# ════════════════════════════════════════════════════════════════════════════
def ensure_chroma_db():
    """Descomprime chroma_db.zip si chroma_db/ no existe"""
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
                st.sidebar.error(f"❌ Error descomprimiendo: {str(e)[:100]}")
                return False
    elif os.path.exists(chroma_dir):
        st.sidebar.success("✅ Base de conocimiento disponible")
        return True
    else:
        st.sidebar.warning("⚠️ Sin base de conocimiento pre-cargada")
        return False

# Descomprimir al inicio (antes de cualquier otra operación)
CHROMA_AVAILABLE = ensure_chroma_db()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE ENTORNO Y API
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
    api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "deepseek/deepseek-v3.2"

# ════════════════════════════════════════════════════════════════════════════
# CARGAR VECTORSTORE CHROMADB (RAG)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_vectorstore():
    """Carga el vectorstore ChromaDB pre-entrenado"""
    if not CHROMA_AVAILABLE:
        return None
    
    persist_dir = "chroma_db"
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error cargando vectorstore: {str(e)[:100]}")
        return None

vectorstore = load_vectorstore()

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

# Cabecera con logos
col_logo1, col_title, col_logo2 = st.columns([1, 2, 1])

with col_logo1:
    try:
        st.image("data/80_anos.png", width=150)
    except:
        st.markdown("### 🎓 80 años")

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
    try:
        st.image("data/univalle_logo.png", width=150)
    except:
        st.markdown("### 🏛️ Univalle")

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

# Estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# Panel lateral con información
with st.sidebar:
    st.markdown("### 📚 Información")
    st.markdown("""
    **ChatAcredita** es un asistente especializado en procesos de acreditación de programas de la Escuela de Ingeniería de Sistemas y Computación.
    
    ### 📥 Cómo usar:
    1. Sube un documento PDF relacionado con acreditación
    2. Escribe tu pregunta en el chat
    3. Obtén respuestas basadas en el documento y la base de conocimiento
    
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
        st.markdown("Sube documentos a la carpeta `pdfs/` y ejecuta `entrenamiento.py`")

# Subida de documento
uploaded_file = st.file_uploader(
    "📄 Sube un documento sobre acreditación (PDF recomendado)",
    type=["pdf", "txt"],
    key="uploader"
)

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            # Procesar PDF con PyMuPDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            # Procesar TXT
            text = uploaded_file.getvalue().decode("utf-8")
        
        # Limitar texto para evitar tokens excesivos
        st.session_state.document_text = text[:8000]
        st.session_state.document_name = uploaded_file.name
        
        st.success(f"✅ Documento procesado")
        st.info(f"📄 **{st.session_state.document_name}** cargado correctamente")
        
    except Exception as e:
        st.error(f"❌ Error al procesar el documento: {str(e)[:150]}")

# Mostrar historial de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Analizando información...")
        
        # Construir contexto para el LLM
        context_sources = []
        full_context = ""
        
        # 1. Recuperar de vectorstore si existe (RAG)
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(prompt, k=3)
                if docs:
                    rag_context = "\n\n".join([doc.page_content for doc in docs])
                    full_context += f"Documentos de referencia:\n{rag_context}\n\n"
                    context_sources.append("Base de conocimiento RAG")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error en búsqueda semántica: {str(e)[:50]}")
        
        # 2. Agregar documento subido por usuario
        if st.session_state.document_text:
            full_context += f"Documento actual ({st.session_state.document_name}):\n{st.session_state.document_text}"
            context_sources.append(f"Documento subido: {st.session_state.document_name}")
        
        # 3. Si no hay contexto, usar mensaje informativo
        if not full_context.strip():
            full_context = "No hay documentos disponibles para responder esta pregunta. Por favor, sube un documento relacionado con acreditación de programas."
            context_sources = ["Sin contexto disponible"]
        
        # Mostrar fuentes utilizadas
        sources_text = " | ".join(context_sources)
        placeholder.markdown(f"🔍 Fuentes: {sources_text}\n\nGenerando respuesta...")
        
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
                max_tokens=800,
                temperature=0.3,
                stream=True
            )
            
            # Mostrar respuesta en streaming
            answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer + "▌")
            
            # Mostrar respuesta final con formato
            final_answer = answer.strip()
            if not final_answer:
                final_answer = "⚠️ No pude generar una respuesta. Por favor, reformula tu pregunta o verifica que el documento contenga información relevante."
            
            placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            error_msg = f"❌ Error al generar respuesta: {str(e)[:200]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Sugerencias de solución
            if "401" in str(e) or "authentication" in str(e).lower():
                st.error("🔑 **Solución:** Verifica que tu API key de OpenRouter esté correctamente configurada en Settings → Secrets")
            elif "404" in str(e) or "not found" in str(e).lower():
                st.error("🤖 **Solución:** El modelo no está disponible. Usa 'mistralai/mistral-7b-instruct' en tu configuración")
            elif "rate limit" in str(e).lower():
                st.error("⏱️ **Solución:** Límite de uso alcanzado. Espera unos minutos o verifica tus créditos en https://openrouter.ai/credits")

# Mensaje inicial si no hay historial
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **Escuela de Ingeniería de Sistemas y Computación**.
        
        ### 🚀 Para empezar:
        1. **Sube un documento** relacionado con acreditación usando el botón de arriba
        2. **Haz tu pregunta** en el campo de chat
        3. **Obtén respuestas** basadas en el contenido de tu documento y mi base de conocimiento
        
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