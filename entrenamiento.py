# entrenamiento_chroma.py - Versión corregida y optimizada para Streamlit Cloud
import os
import shutil
import time
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import pymupdf4llm
import pickle

def clean_chroma_db(persist_directory="chroma_db"):
    """Elimina completamente el directorio chroma_db existente"""
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"✅ Carpeta '{persist_directory}' eliminada")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar {persist_directory}: {e}")

def _process_pdfs():
    """Procesa PDFs con pymupdf4llm (conserva estructura) con barra de progreso"""
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    pdf_folder = "pdfs"
    if not os.path.exists(pdf_folder):
        print(f"❌ Carpeta '{pdf_folder}' no existe. Crea la carpeta y agrega tus PDFs de acreditación.")
        return [], []
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No hay PDFs en '{pdf_folder}'. Agrega documentos de acreditación.")
        return [], []
    
    all_chunks = []
    total_chunks = 0
    total_pdfs = len(pdf_files)
    
    print(f"\n📄 Procesando {total_pdfs} PDFs con pymupdf4llm...")
    print("=" * 60)
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            # Barra de progreso para PDFs
            progress = int((idx / total_pdfs) * 30)
            bar = "█" * progress + "░" * (30 - progress)
            print(f"\rPDF {idx}/{total_pdfs} [{bar}] {pdf_file}", end="", flush=True)
            
            doc = fitz.open(os.path.join(pdf_folder, pdf_file))
            text = pymupdf4llm.to_markdown(doc)
            doc.close()
            
            if text.strip():
                chunks = text_splitter.split_text(text)
                valid_chunks = 0
                for chunk in chunks:
                    if len(chunk.strip()) > 100:
                        doc_obj = Document(
                            page_content=chunk.strip(), 
                            metadata={"source": pdf_file}
                        )
                        docs.append(doc_obj)
                        all_chunks.append(chunk.strip())
                        total_chunks += 1
                        valid_chunks += 1
                print(f" → ✅ {valid_chunks} chunks válidos", flush=True)
            else:
                print(f" → ⚠️  texto vacío", flush=True)
        except Exception as e:
            print(f" → ❌ Error: {str(e)[:80]}", flush=True)
    
    print("\n" + "=" * 60)
    print(f"📊 Total de chunks generados: {total_chunks}")
    return docs, all_chunks

def _create_index(persist_directory="chroma_db"):
    """Crea índice ChromaDB con modelo compatible y barra de progreso"""
    print("🧹 Limpiando directorio chroma_db...")
    clean_chroma_db(persist_directory)
    
    print("\n" + "=" * 60)
    print("🚀 GENERADOR DE VECTORSTORE - CHATAACREDITA")
    print("=" * 60)
    
    docs, _ = _process_pdfs()
    
    if not docs:
        print("\n❌ No se generaron documentos válidos")
        return False
    
    print(f"\n🔍 Creando índice ChromaDB con {len(docs)} chunks...")
    print("=" * 60)
    
    try:
        # ✅ Modelo ligero y compatible: bge-small-en-v1.5 (384d)
        print("📥 Cargando modelo de embeddings: BAAI/bge-m3...")
        start_time = time.time()
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        model_load_time = time.time() - start_time
        print(f"✅ Modelo cargado en {model_load_time:.1f}s")
        
        # Crear vectorstore
        print("\n🧠 Generando embeddings y creando índice...")
        index_start = time.time()
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Simular progreso visual (ChromaDB no expone callbacks reales)
        for i in range(30):
            time.sleep(0.03)
            progress = "█" * (i + 1) + "░" * (29 - i)
            print(f"\r    [{progress}] {int((i+1)/30*100)}%", end="", flush=True)
        print("\r    [██████████████████████████████] 100%", flush=True)
        
        index_time = time.time() - index_start
        print(f"✅ Índice creado en {index_time:.1f}s")
        
        # ✅ Guardar documentos para búsqueda lexical simple (sin BM25Retriever problemático)
        # Usamos rank-bm25 en app.py para evitar conflictos de versiones de LangChain
        docs_path = os.path.join(persist_directory, "documents.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"✅ Documentos guardados para búsqueda lexical en {docs_path}")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ ¡VECTORSTORE CREADO EXITOSAMENTE!")
        print("=" * 60)
        print(f"📁 Ubicación: {persist_directory}/")
        print(f"📚 Modelo de embeddings: BAAI/bge-small-en-v1.5 (384d)")
        print(f"📄 Documentos indexados: {len(docs)} chunks")
        print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error creando índice: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = _create_index()
    
    if success:
        print("\n📌 PRÓXIMOS PASOS:")
        print("1. Comprime la carpeta chroma_db:")
        print("   Compress-Archive -Path 'chroma_db' -DestinationPath 'chroma_db.zip'")
        print("2. Verifica tamaño (< 100 MB):")
        print("   (Get-Item chroma_db.zip).Length / 1MB")
        print("3. Sube chroma_db.zip a GitHub junto con app.py")
        print("4. Despliega en Streamlit Cloud")
        print("\n🎉 ¡Tu sistema RAG está listo para usar!")
    else:
        print("\n" + "=" * 60)
        print("❌ FALLÓ LA CREACIÓN DEL VECTORSTORE")
        print("=" * 60)
        print("\n🔍 SOLUCIÓN DE PROBLEMAS:")
        print("- Verifica que la carpeta 'pdfs/' exista y contenga PDFs válidos")
        print("- Asegúrate de tener conexión a internet (para descargar embeddings)")
        print("- Ejecuta: pip install -U langchain-huggingface sentence-transformers")
        print("- Si el error persiste, elimina la carpeta chroma_db y reintenta")