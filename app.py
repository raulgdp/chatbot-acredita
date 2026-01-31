import os
import streamlit as st
from openai import OpenAI

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
#MODEL = "mistralai/mistral-7b-instruct"  # ✅ Modelo válido y gratuito
MODEL ="deeseek/deepsek-v3.2"

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#c00000;'>🤖 ChatAcredita</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center;color:#1a5276;margin-bottom:20px;'>"
    "Asistente de Acreditación - EISC Univalle</h3>",
    unsafe_allow_html=True
)

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

# Input del usuario (ÚNICO lugar donde se modifica session_state.messages)
if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Analizando...")
        
        context = st.session_state.document_text if st.session_state.document_text else "No hay documento cargado."
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde SOLO con base en el documento proporcionado."},
                    {"role": "user", "content": f"Documento:\n{context}\n\nPregunta: {prompt}"}
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

# Mensaje de bienvenida si no hay historial
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente especializado en procesos de acreditación de programas de la **Escuela de Ingeniería de Sistemas y Computación**.
        
        **Para empezar:**
        1. Sube un documento PDF relacionado con acreditación
        2. Escribe tu pregunta en el chat
        3. Obtén respuestas basadas en tu documento
        
        *Ejemplo: "¿Cuáles son los requisitos para acreditar un programa de pregrado?"*
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>"
    "Desarrollado por GUIA - EISC Univalle</div>",
    unsafe_allow_html=True
)