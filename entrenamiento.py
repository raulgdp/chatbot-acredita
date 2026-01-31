# entrenamiento_chroma.py - Versión optimizada para Streamlit Cloud
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
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
    """Procesa PDFs con pymupdf4llm (conserva estructura)"""
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
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(os.path.join(pdf_folder, pdf_file))
            text = pymupdf4llm.to_markdown(doc)
            doc.close()
            
            if text.strip():
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    # Limpiar chunks muy cortos o sin significado
                    if len(chunk.strip()) > 100:
                        doc_obj = Document(
                            page_content=chunk.strip(), 
                            metadata={"source": pdf_file}
                        )
                        docs.append(doc_obj)
                        all_chunks.append(chunk.strip())
                        total_chunks += 1
                print(f"✅ {pdf_file}: {len(chunks)} chunks ({total_chunks} total)")
            else:
                print(f"⚠️ {pdf_file}: texto vacío o no procesable")
        except Exception as e:
            print(f"❌ Error procesando {pdf_file}: {str(e)[:120]}")
    
    print(f"\n📊 Total de chunks generados: {total_chunks}")
    return docs, all_chunks

def _create_index(persist_directory="chroma_db"):
    """Crea índice ChromaDB + BM25 con modelo compatible"""
    print("🧹 Limpiando directorio chroma_db...")
    clean_chroma_db(persist_directory)
    
    print("\n📄 Procesando PDFs con pymupdf4llm...")
    docs, _ = _process_pdfs()
    
    if not docs:
        print("❌ No se generaron documentos válidos")
        return False
    
    print(f"\n🔍 Creando índice ChromaDB con {len(docs)} chunks...")
    try:
        # ✅ Modelo ligero y compatible: bge-small-en-v1.5 (384d)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Crear vectorstore desde cero
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # ✅ Crear BM25 retriever (búsqueda por palabras clave)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5
        
        # ✅ Guardar BM25 para usar en app.py
        bm25_path = os.path.join(persist_directory, "bm25_retriever.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        
        print(f"\n✅ Índice guardado en '{persist_directory}/'")
        print(f"✅ Modelo de embeddings: BAAI/bge-small-en-v1.5 (384d)")
        print(f"✅ BM25 retriever guardado en {bm25_path}")
        print(f"✅ Total de documentos indexados: {len(docs)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error creando índice: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🚀 Generador de Vectorstore - ChatAcredita")
    print("="*60)
    success = _create_index()
    
    if success:
        print("\n" + "="*60)
        print("🎉 ¡Vectorstore creado exitosamente!")
        print("="*60)
        print("\n📌 Próximos pasos:")
        print("1. Comprime la carpeta chroma_db:")
        print("   Compress-Archive -Path 'chroma_db' -DestinationPath 'chroma_db.zip'")
        print("2. Sube chroma_db.zip a GitHub junto con app.py")
        print("3. Despliega en Streamlit Cloud")
    else:
        print("\n" + "="*60)
        print("❌ Falló la creación del vectorstore")
        print("="*60)
        print("\n🔍 Solución de problemas:")
        print("- Verifica que la carpeta 'pdfs/' exista y contenga PDFs válidos")
        print("- Asegúrate de tener conexión a internet (para descargar embeddings)")
        print("- Ejecuta: pip install -U langchain-huggingface sentence-transformers")