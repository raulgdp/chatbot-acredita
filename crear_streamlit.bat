@echo off
echo Creando carpeta .streamlit...
mkdir .streamlit 2>nul

echo Creando config.toml...
(
echo [theme]
echo primaryColor = "#c00000"
echo backgroundColor = "#ffffff"
echo secondaryBackgroundColor = "#f0f2f6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo headless = true
echo port = 8501
echo.
echo [browser]
echo gatherUsageStats = false
) > .streamlit\config.toml

echo Creando secrets.toml.example...
(
echo # EJEMPLO - NO subas este archivo con tus claves reales a GitHub
echo # Copia este archivo a secrets.toml y configura tus claves
echo.
echo OPENAI_API_KEY = "sk-or-v1-tu-api-key-aqui"
echo OPENAI_API_BASE = "https://openrouter.ai/api/v1"
) > .streamlit\secrets.toml.example

echo.
echo ✅ Carpeta .streamlit creada correctamente
echo Archivos creados:
dir /b .streamlit
pause