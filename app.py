# app.py - Versión optimizada para Streamlit Cloud
import os
import tempfile
import cv2
import pytesseract
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
import fitz  # PyMuPDF

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PARA STREAMLIT CLOUD
# ════════════════════════════════════════════════════════════════════════════

def is_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False

TESSERACT_AVAILABLE = is_tesseract_available()

# ✅ STREAMLIT CLOUD: Detectar entorno y usar Secrets
IS_STREAMLIT_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_STREAMLIT_CLOUD:
    # En Streamlit Cloud, usar Secrets
    if "OPENAI_API_KEY" in st.secrets:
        API_KEY = st.secrets["OPENAI_API_KEY"]
        API_BASE = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    else:
        st.error("❌ Error: No se encontró OPENAI_API_KEY en Secrets")
        st.stop()
else:
    # En desarrollo local, usar variables de entorno
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# ✅ CORREGIDO: URL SIN ESPACIOS
API_BASE = API_BASE.strip()

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# ✅ CORREdeepseek/deepseek-v3.2Router
MODEL_NAME = "mistralai/mistral-7b-instruct"  # ✅ Gratuito y disponible
# Alternativas válidas:
# "google/gemma-2-9b-it"
# "qwen/qwen-2.5-72b-instruct"
# "anthropic/claude-3.5-haiku"

VECTORSTORE_DIR = "data/vectorstore"  # ✅ Ruta relativa para Streamlit Cloud

# ════════════════════════════════════════════════════════════════════════════
# CACHÉ Y FUNCIONES
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_components():
    try:
        from entrenamiento import load_hybrid_retriever
        vector_store, bm25_retriever = load_hybrid_retriever(VECTORSTORE_DIR)
        if vector_store is None or bm25_retriever is None:
            return None, None, None
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        return vector_store, bm25_retriever, reranker
    except Exception as e:
        st.error(f"❌ Error al cargar componentes: {e}")
        return None, None, None

def hybrid_retrieve_and_rerank(query, vector_store, bm25_retriever, reranker, top_k=5):
    try:
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_store.max_marginal_relevance_search(query, k=10, fetch_k=30)
        combined = list({doc.page_content: doc for doc in bm25_docs + vector_docs}.values())[:10]
        if not combined:
            return [], ""
        pairs = [[query, doc.page_content] for doc in combined]
        scores = reranker.predict(pairs)
        scored = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored[:top_k]]
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        return reranked_docs, context
    except Exception as e:
        st.error(f"Error en recuperación: {e}")
        return [], ""

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="🤖 ChatbotAcredita", page_icon="🎓", layout="wide")

# Logos
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])

with col_logo1:
    try:
        st.image("data/logo2.png", width=180)
    except:
        st.warning("⚠️ Logo no encontrado: data/logo2.png")

with col_logo2:
    st.markdown(
        '<div style="text-align: center; padding: 15px 0;">'
        '<h3 style="color: #c00000; margin: 5px 0; font-weight: bold; font-size: 1.4rem;">'
        'Escuela de Ingeniería de Sistemas y Computación</h3>'
        '<h4 style="color: #1a5276; margin: 3px 0; font-size: 1.15rem;">'
        'Chatbot de Acreditación de Programas</h4>'
        '</div>',
        unsafe_allow_html=True
    )

with col_logo3:
    try:
        st.image("data/univalle_logo.png", width=180)
    except:
        st.warning("⚠️ Logo no encontrado: data/univalle_logo.png")

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

st.title("🤖 ChatAcredita")
st.markdown("Asistente especializado en procesos de acreditación de programas de la EISC. Escribe **'salir'** para despedirte.")

# Mensaje de bienvenida
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **Escuela de Ingeniería de Sistemas y Computación**.
        
        Puedo ayudarte con:
        - 📋 Requisitos para acreditación de programas
        - 📚 Documentación necesaria
        - 🎯 Procesos y procedimientos
        - 📊 Estándares de calidad
        
        **💡 Tip:** Puedes subir documentos PDF relacionados con acreditación para obtener respuestas más precisas.
        
        ¿En qué puedo ayudarte hoy?
        """)

if not TESSERACT_AVAILABLE:
    st.info("ℹ️ Tesseract OCR no está disponible en esta plataforma.")

# Inicializar estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# Cargar componentes
vector_store, bm25_retriever, reranker = load_components()
if vector_store is None or bm25_retriever is None or reranker is None:
    st.error("❌ No se pudieron cargar los índices. Verifica que 'data/vectorstore' exista.")
    st.stop()

# Layout
col_pdf, col_chat = st.columns([1, 2])

with col_pdf:
    st.subheader("📄 Visor de Documentos")
    uploaded_file_widget = st.file_uploader(
        "Sube un PDF, TXT o imagen relacionado con acreditación",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        key="file_uploader"
    )

    if uploaded_file_widget is not None:
        current = st.session_state.get("uploaded_file", None)
        if (current is None or 
            current.name != uploaded_file_widget.name or 
            current.size != uploaded_file_widget.size):
            st.session_state.uploaded_file = uploaded_file_widget
            if uploaded_file_widget.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file_widget.getvalue())
                    tmp_path = tmp.name
                doc = fitz.open(tmp_path)
                st.session_state.pdf_pages = [page.get_pixmap() for page in doc]
                doc.close()
                os.unlink(tmp_path)
                st.session_state.current_page = 0
    else:
        st.session_state.uploaded_file = None
        st.session_state.pdf_pages = []
        st.session_state.current_page = 0

    if st.session_state.uploaded_file is not None:
        file = st.session_state.uploaded_file
        if file.type == "application/pdf" and st.session_state.pdf_pages:
            pages = st.session_state.pdf_pages
            total_pages = len(pages)
            col_prev, col_center, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("◀️ Anterior") and st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
            with col_center:
                st.markdown(f"<div style='text-align: center;'>Página {st.session_state.current_page + 1} de {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.button("Siguiente ▶️") and st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page += 1
            img_data = pages[st.session_state.current_page].tobytes("png")
            st.image(img_data, width=600)
        elif file.type.startswith("image/"):
            st.image(file, caption=file.name, width=600)
        elif file.type == "text/plain":
            content = file.getvalue().decode("utf-8")
            st.text_area("Contenido del archivo", content, height=400)

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.session_state.get("user_prompt", None)

    PALABRAS_SALIDA = {"salir", "cerrar", "adiós", "chao", "hasta", "luego", "gracias", "fin", "terminar"}

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        palabras_usuario = {word.strip().lower() for word in prompt.split()}
        if palabras_usuario & PALABRAS_SALIDA:
            farewell_msg = "👋 ¡Hasta pronto! Si necesitas ayuda con procesos de acreditación, ¡regresa cuando quieras!"
            with st.chat_message("assistant"):
                st.markdown(farewell_msg)
            st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
            st.info("💡 Puedes cerrar esta pestaña o seguir chateando cuando lo necesites.")
        else:
            extra_context = ""
            temp_sources = []
            uploaded_file_to_use = st.session_state.uploaded_file

            if uploaded_file_to_use is not None:
                try:
                    file_ext = os.path.splitext(uploaded_file_to_use.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                        tmp_file.write(uploaded_file_to_use.getvalue())
                        tmp_path = tmp_file.name

                    md_text = ""
                    if uploaded_file_to_use.type == "application/pdf":
                        md_text = pymupdf4llm.to_markdown(tmp_path)
                    elif uploaded_file_to_use.type == "text/plain":
                        md_text = uploaded_file_to_use.getvalue().decode("utf-8")
                    elif uploaded_file_to_use.type in ["image/png", "image/jpeg", "image/jpg"]:
                        if not TESSERACT_AVAILABLE:
                            md_text = ""
                        else:
                            image = cv2.imread(tmp_path)
                            if image is None:
                                raise ValueError("No se pudo cargar la imagen.")
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            thresh = cv2.adaptiveThreshold(
                                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                            )
                            custom_config = r'--oem 3 --psm 6 -l spa+eng'
                            md_text = pytesseract.image_to_string(thresh, config=custom_config)
                    os.unlink(tmp_path)

                    if md_text.strip():
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(md_text)
                        extra_context = "\n\n".join(chunks)
                        temp_sources.append(f"📄 {uploaded_file_to_use.name}")
                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: {e}")

            docs, base_context = hybrid_retrieve_and_rerank(prompt, vector_store, bm25_retriever, reranker)
            full_context = base_context
            if extra_context:
                full_context = extra_context + "\n\n" + (base_context if base_context else "")

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🧠 Analizando información de acreditación...")

                if not full_context.strip():
                    no_docs_msg = "⚠️ No encontré información relevante sobre acreditación en los documentos cargados. ¿Podrías reformular tu pregunta o cargar documentos relacionados con procesos de acreditación de la EISC?"
                    placeholder.markdown(no_docs_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})
                else:
                    messages = [
                        {"role": "system", "content": "Eres ChatAcredita, un asistente especializado en procesos de acreditación de programas de la Escuela de Ingeniería de Sistemas y Computación de la Universidad del Valle. Responde con información precisa, basada únicamente en el contexto proporcionado. Sé claro, conciso y profesional. Si no sabes algo, indícalo honestamente."},
                        {"role": "user", "content": f"Contexto:\n{full_context}\n\nPregunta sobre acreditación: {prompt}\n\nRespuesta:"}
                    ]

                    try:
                        stream = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            max_tokens=1024,
                            temperature=0.3,
                            stream=True
                        )

                        answer = ""
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                answer += chunk.choices[0].delta.content
                                placeholder.markdown(answer + "▌")

                        formatted_answer = f'<span style="color:#2c3e50;font-weight:500;">{answer}</span>'
                        all_sources = set(temp_sources)
                        if docs:
                            all_sources.update({os.path.basename(d.metadata.get("source", "Desconocido")) for d in docs})
                        if all_sources:
                            formatted_answer += f'<br><br><span style="color:#7f8c8d;font-size:0.92em;">🎓 Fuentes: {", ".join(all_sources)}</span>'

                        placeholder.markdown(formatted_answer, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

                    except Exception as e:
                        error_detail = str(e)[:300]
                        if "404" in error_detail or "Not Found" in error_detail:
                            error_msg = "❌ Error: Modelo no encontrado. Usa un modelo válido de OpenRouter."
                        elif "401" in error_detail or "authentication" in error_detail.lower():
                            error_msg = "❌ Error: API Key inválida o sin créditos en OpenRouter."
                        else:
                            error_msg = f"❌ Error: {error_detail}"
                        
                        placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    elif st.session_state.uploaded_file is not None:
        st.info("📌 Documento cargado. Escribe tu pregunta sobre acreditación para ayudarte.")

st.chat_input("Ej: ¿Qué programas se pueden acreditar en la EISC?", key="user_prompt")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">'
    'Desarrollado por GUIA - Grupo de Univalle en Inteligencia Artificial | '
    'Escuela de Ingeniería de Sistemas y Computación | Universidad del Valle'
    '</div>',
    unsafe_allow_html=True
)