# app.py - Versión 100% compatible con Streamlit Cloud (SIN FAISS)
import os
import tempfile
import cv2
import pytesseract
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PARA STREAMLIT CLOUD
# ════════════════════════════════════════════════════════════════════════════
IS_STREAMLIT_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_STREAMLIT_CLOUD:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def is_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        return False

TESSERACT_AVAILABLE = is_tesseract_available()

# API Configuration
if IS_STREAMLIT_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ ERROR: Configura OPENAI_API_KEY en Settings → Secrets")
        st.stop()
    API_KEY = st.secrets["OPENAI_API_KEY"]
    API_BASE = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
    if not API_KEY:
        st.warning("⚠️ API key no configurada. Usa variables de entorno OPENAI_API_KEY")
        st.stop()

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
MODEL_NAME = "mistralai/mistral-7b-instruct"

# ════════════════════════════════════════════════════════════════════════════
# RECUPERACIÓN SIMPLE SIN FAISS (USANDO BÚSQUEDA BÁSICA)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_reranker():
    try:
        return CrossEncoder("BAAI/bge-reranker-base")
    except:
        return None

def simple_retrieve(text_chunks, query, top_k=3):
    """Búsqueda simple por coincidencia de palabras clave"""
    query_words = set(query.lower().split())
    scored = []
    for chunk in text_chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="🤖 ChatbotAcredita", page_icon="🎓", layout="wide")

col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])

with col_logo1:
    try:
        st.image("data/logo2.png", width=180)
    except:
        st.markdown("### 🎓 EISC")

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
        st.markdown("### 🏛️ Univalle")

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)

st.title("🤖 ChatAcredita")
st.markdown("Asistente especializado en procesos de acreditación de programas de la EISC. Escribe **'salir'** para despedirte.")

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **Escuela de Ingeniería de Sistemas y Computación**.
        
        **💡 Cómo usar este chatbot:**
        1. Sube un documento PDF/TXT relacionado con acreditación en el panel izquierdo
        2. Escribe tu pregunta en el chat
        3. ¡Obtén respuestas instantáneas basadas en tu documento!
        
        *Nota: No tengo conocimiento previo de documentos de acreditación. Necesito que subas tus documentos para ayudarte.*
        """)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# Cargar reranker (opcional)
reranker = load_reranker()

col_pdf, col_chat = st.columns([1, 2])

with col_pdf:
    st.subheader("📄 Sube tu documento")
    uploaded_file_widget = st.file_uploader(
        "PDF, TXT o imagen sobre acreditación",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        key="file_uploader"
    )

    if uploaded_file_widget is not None:
        current = st.session_state.get("uploaded_file", None)
        if (current is None or current.name != uploaded_file_widget.name):
            st.session_state.uploaded_file = uploaded_file_widget
            st.session_state.document_chunks = []
            
            try:
                if uploaded_file_widget.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file_widget.getvalue())
                        tmp_path = tmp.name
                    doc = fitz.open(tmp_path)
                    text = ""
                    st.session_state.pdf_pages = [page.get_pixmap() for page in doc]
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    os.unlink(tmp_path)
                    
                    # Dividir en chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    st.session_state.document_chunks = splitter.split_text(text)
                    st.success(f"✅ PDF procesado ({len(st.session_state.document_chunks)} fragmentos)")
                    
                elif uploaded_file_widget.type == "text/plain":
                    text = uploaded_file_widget.getvalue().decode("utf-8")
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    st.session_state.document_chunks = splitter.split_text(text)
                    st.success(f"✅ TXT procesado ({len(st.session_state.document_chunks)} fragmentos)")
                    
                elif uploaded_file_widget.type.startswith("image/"):
                    if TESSERACT_AVAILABLE:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(uploaded_file_widget.getvalue())
                            tmp_path = tmp.name
                        image = cv2.imread(tmp_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 -l spa+eng')
                        os.unlink(tmp_path)
                        
                        if text.strip():
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            st.session_state.document_chunks = splitter.split_text(text)
                            st.success(f"✅ Imagen procesada ({len(st.session_state.document_chunks)} fragmentos)")
                        else:
                            st.warning("⚠️ No se detectó texto en la imagen")
                    else:
                        st.warning("⚠️ OCR no disponible en esta plataforma")
            except Exception as e:
                st.error(f"❌ Error al procesar archivo: {str(e)[:100]}")

    # Visor de PDF si existe
    if st.session_state.uploaded_file and st.session_state.uploaded_file.type == "application/pdf" and st.session_state.pdf_pages:
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

        if {word.strip().lower() for word in prompt.split()} & PALABRAS_SALIDA:
            farewell_msg = "👋 ¡Hasta pronto! Si necesitas ayuda con procesos de acreditación, ¡regresa cuando quieras!"
            with st.chat_message("assistant"):
                st.markdown(farewell_msg)
            st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🧠 Analizando tu documento...")
                
                # Obtener contexto del documento subido
                if st.session_state.document_chunks:
                    relevant_chunks = simple_retrieve(st.session_state.document_chunks, prompt, top_k=3)
                    context = "\n\n".join(relevant_chunks)
                else:
                    context = ""
                
                if not context.strip():
                    msg = "⚠️ No tengo documentos cargados para responder tu pregunta. Por favor, sube un PDF/TXT sobre acreditación en el panel izquierdo."
                    placeholder.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    messages = [
                        {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con base en el documento proporcionado. Sé preciso y conciso."},
                        {"role": "user", "content": f"Documento:\n{context}\n\nPregunta: {prompt}\n\nRespuesta:"}
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
                        
                        placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"❌ Error de API: {str(e)[:150]}"
                        placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    elif st.session_state.uploaded_file is not None:
        st.info("📌 Documento cargado. Escribe tu pregunta sobre acreditación.")

st.chat_input("Ej: ¿Cuáles son los requisitos para acreditar un programa?", key="user_prompt")

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">'
    'Desarrollado por GUIA - Grupo de Univalle en Inteligencia Artificial | '
    'Escuela de Ingeniería de Sistemas y Computación | Universidad del Valle'
    '</div>',
    unsafe_allow_html=True
)