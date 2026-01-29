# entrenamiento.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import pymupdf4llm

def load_hybrid_retriever(vectorstore_dir="vectorstore"):
    """
    Carga los índices para la interfaz de Streamlit.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vector_store = FAISS.load_local(
            vectorstore_dir,
            embeddings,
            index_name="faiss_index",
            allow_dangerous_deserialization=True
        )
        docs = list(vector_store.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5
        return vector_store, bm25_retriever
    except Exception as e:
        print(f"❌ Error al cargar índices: {e}")
        return None, None

def _process_pdfs():
    """Procesa PDFs y devuelve documentos."""
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for pdf_file in os.listdir("pdfs"):
        if pdf_file.endswith(".pdf"):
            try:
                doc = fitz.open(f"pdfs/{pdf_file}")
                text = pymupdf4llm.to_markdown(doc)
                doc.close()
                if text.strip():
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        docs.append(Document(page_content=chunk, metadata={"source": pdf_file}))
                    print(f"✅ {pdf_file}: {len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Error en {pdf_file}: {e}")
    return docs

def _create_index():
    """Crea y guarda el índice FAISS."""
    print("📄 Procesando PDFs...")
    docs = _process_pdfs()
    
    if not docs:
        print("❌ No se generaron documentos")
        return
    
    print(f"🔍 Creando índice con {len(docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = FAISS.from_documents(docs, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vector_store.save_local("vectorstore", index_name="faiss_index")
    print("✅ Índice guardado en 'vectorstore/'")

# Ejecutar creación de índice cuando se llama directamente
if __name__ == "__main__":
    _create_index()