import os
import streamlit as st
from openai import OpenAI
import fitz

IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ ERROR: Configura OPENAI_API_KEY en Settings → Secrets")
        st.stop()
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1  ").strip()
else:
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    api_base = "https://openrouter.ai/api/v1  "

client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "deepseek/deepseek-v3.2"

st.set_page_config(page_title="ChatAcredita", page_icon="🎓", layout="wide")

# ════════════════════════════════════════════════════════════════════════════
# CABECERA CON LOGOS (MODIFICADO)
# ════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 3])

with col_logo:
    # Reemplaza 'data/' con la ruta correcta si tus logos están en otra carpeta
    try:
        st.image("data/80_anos.png", width=180)
    except Exception as e:
        st.markdown("<h3 style='text-align:center;color:#c00000;'>80 años<br>Construyendo Futuro</h3>", unsafe_allow_html=True)

with col_title:
    st.markdown("<h1 style='text-align:center;color:#c00000;'>🤖 ChatAcredita</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#1a5276;'>Asistente de Acreditación - EISC Univalle</h3>", unsafe_allow_html=True)

st.markdown('<hr style="border: 2px solid #c00000; margin: 10px 0;">', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded = st.file_uploader("📄 Sube un PDF sobre acreditación", type=["pdf"])

if uploaded:
    try:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        st.session_state.document_text = text[:5000]
        st.success(f"✅ PDF procesado ({min(len(text), 5000)} caracteres)")
    except Exception as e:
        st.error(f"❌ Error al procesar PDF: {str(e)[:100]}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Analizando...")
        
        context = st.session_state.document_text if st.session_state.document_text else "No hay documento cargado."
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres ChatAcredita, asistente de acreditación de la EISC. Responde basado SOLO en el documento proporcionado."},
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
            error_msg = f"⚠️ Error: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>Desarrollado por GUIA - EISC Univalle</div>", unsafe_allow_html=True)