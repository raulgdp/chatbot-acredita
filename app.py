import os
import streamlit as st
from openai import OpenAI
import fitz

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
# CONFIGURACIÓN DE API - MODELO VÁLIDO Y SECRETS
# ════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.getenv("HOME") == "/home/appuser"

if IS_CLOUD:
    # ✅ Verificación EXPLÍCITA de Secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("""
        ❌ ERROR CRÍTICO: OPENAI_API_KEY no configurado en Secrets
        
        🔑 Solución:
        1. Ve a https://share.streamlit.io/raulgdp/chatbot-acredita
        2. Click en "⋮" → Settings → Secrets
        3. Agrega:
           OPENAI_API_KEY = "sk-or-v1-tu-api-key-real"
           OPENAI_API_BASE = "https://openrouter.ai/api/v1"
        """)
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
else:
    # Modo local (desarrollo)
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    api_base = "https://openrouter.ai/api/v1".strip()

# ✅ MODELO VÁLIDO DE DEEPSEEK (deepseek-v3.2 NO EXISTE)
MODEL = "deepseek/deepseek-chat"  # ✅ ÚNICO modelo DeepSeek válido en OpenRouter

try:
    client = OpenAI(api_key=api_key, base_url=api_base)
except Exception as e:
    st.error(f"""
    ❌ Error al inicializar OpenAI:
    {str(e)[:200]}
    
    🔑 Posibles causas:
    • API key inválida o expirada
    • Límite de créditos alcanzado en OpenRouter
    • Base URL incorrecta
    
    Verifica tu key en: https://openrouter.ai/keys
    """)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO MÍNIMA (GARANTIZADA PARA FUNCIONAR)
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

uploaded = st.file_uploader("📄 Sube un PDF sobre acreditación", type=["pdf"])

if uploaded:
    try:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)[:5000]
        doc.close()
        st.session_state.document_text = text
        st.session_state.document_name = uploaded.name
        st.success(f"✅ PDF procesado: {st.session_state.document_name}")
    except Exception as e:
        st.error(f"❌ Error al procesar PDF: {str(e)[:100]}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre acreditación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Analizando tu documento...")
        
        context = st.session_state.document_text if st.session_state.document_text else "No hay documento cargado."
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,  # ✅ deepseek/deepseek-chat (VÁLIDO)
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres ChatAcredita, asistente de acreditación de la EISC. "
                            "Responde SOLO con base en el documento proporcionado. "
                            "Sé preciso y conciso."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Documento:\n{context}\n\nPregunta: {prompt}"
                    }
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
            error_msg = f"❌ Error API: {str(e)[:150]}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Diagnóstico específico para errores comunes
            error_str = str(e).lower()
            if "404" in error_str and "model" in error_str:
                st.error("""
                🔑 **ERROR DE MODELO:**
                El modelo 'deepseek-v3.2' NO EXISTE en OpenRouter.
                
                ✅ Usa SOLO estos modelos válidos:
                • deepseek/deepseek-chat (recomendado)
                • deepseek/deepseek-chat:free (gratuito)
                
                Lista completa: https://openrouter.ai/models
                """)
            elif "401" in error_str or "unauthorized" in error_str:
                st.error("""
                🔑 **ERROR DE AUTENTICACIÓN:**
                API key inválida o sin créditos.
                
                ✅ Solución:
                1. Regenera tu key en https://openrouter.ai/keys
                2. Configura Secrets en Streamlit Cloud con la nueva key
                """)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 ¡Hola! Soy **ChatAcredita**, tu asistente de acreditación de la **EISC**.
        
        **Para empezar:**
        1. Sube un documento PDF relacionado con acreditación
        2. Escribe tu pregunta en el chat
        3. Obtén respuestas basadas SOLO en tu documento
        
        *Ejemplo: "¿Cuáles son los requisitos para acreditar un programa de pregrado?"*
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:0.9em;'>"
    "Desarrollado por GUIA - EISC Univalle</div>",
    unsafe_allow_html=True
)