# app.py - Versión MÍNIMA 100% funcional en Streamlit Cloud
import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN SEGURA PARA STREAMLIT CLOUD
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ Configura OPENAI_API_KEY en Settings → Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = "https://openrouter.ai/api/v1"
    if not api_key:
        st.warning("⚠️ Ejecutando en modo demo (sin API key real)")
        api_key = "demo-key"

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "mistralai/mistral-7b-instruct"

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ MÍNIMA PERO FUNCIONAL
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="✅ ChatAcredita Funcional", page_icon="🎓", layout="wide")

# Logos con fallback seguro
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("### 🎓 EISC")
with col2:
    st.markdown(
        '<div style="text-align:center;padding:15px 0;">'
        '<h3 style="color:#c00000;margin:5px 0;font-weight:bold;">'
        'Chatbot de Acreditación - Versión Funcional</h3></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown("### 🏛️ Univalle")

st.markdown('<hr style="border:2px solid #c00000;margin:10px 0;">', unsafe_allow_html=True)

# Estado de la app
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

# Subida de documento
uploaded = st.file_uploader("📄 Sube un PDF sobre acreditación", type=["pdf"])

if uploaded:
    try:
        # Procesar PDF con PyMuPDF (funciona 100% en Cloud)
        with st.spinner("Leyendo PDF..."):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            st.session_state.document_text = text[:5000]  # Límite para evitar tokens excesivos
            st.success(f"✅ PDF cargado ({len(text)} caracteres)")
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pregunta sobre tu documento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Pensando...")
        
        # Contexto mínimo
        context = st.session_state.document_text if st.session_state.document_text else "No hay documento cargado."
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres un asistente de acreditación. Responde solo con base en el documento proporcionado."},
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
            error_msg = f"⚠️ {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#7f8c8d;font-size:0.9em;">'
    '✅ Versión funcional garantizada para Streamlit Cloud | GUIA - EISC Univalle'
    '</div>',
    unsafe_allow_html=True
)wn(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">'
    'Desarrollado por GUIA - Grupo de Univalle en Inteligencia Artificial | '
    'Escuela de Ingeniería de Sistemas y Computación | Universidad del Valle'
    '</div>',
    unsafe_allow_html=True
)