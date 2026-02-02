# entrenamiento_qdrant_bm25.py - Vectorstore Qdrant + BM25 con bge-small-en-v1.5
import os
import shutil
import uuid
import pickle
import time
import torch

# ✅ VERIFICAR GPU
print("=" * 70)
print("🚀 VERIFICANDO GPU PARA BGE-SMALL")
print("=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda"
else:
    print("⚠️  GPU NO detectada. Usando CPU...")
    DEVICE = "cpu"
print("=" * 70)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import pymupdf4llm

def clean_qdrant_db(collection_path="qdrant_db"):
    """Limpia directorio Qdrant existente"""
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)
    os.makedirs(collection_path, exist_ok=True)
    print(f"✅ Directorio '{collection_path}' preparado")

def process_pdfs(pdf_folder="pdfs", chunk_size=1000, chunk_overlap=200):
    """Extrae texto de PDFs y divide en chunks"""
    if not os.path.exists(pdf_folder):
        print(f"❌ Carpeta '{pdf_folder}' no existe. Crea la carpeta y agrega tus PDFs.")
        return [], []
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No hay PDFs en '{pdf_folder}'.")
        return [], []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    sources = []
    total_chunks = 0
    
    print(f"\n📄 Procesando {len(pdf_files)} PDFs...")
    print("=" * 60)
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(os.path.join(pdf_folder, pdf_file))
            text = pymupdf4llm.to_markdown(doc)
            doc.close()
            
            if text.strip():
                file_chunks = splitter.split_text(text)
                valid_chunks = [c.strip() for c in file_chunks if len(c.strip()) > 100]
                chunks.extend(valid_chunks)
                sources.extend([pdf_file] * len(valid_chunks))
                total_chunks += len(valid_chunks)
                print(f"✅ {pdf_file}: {len(valid_chunks)} chunks")
            else:
                print(f"⚠️ {pdf_file}: texto vacío")
        except Exception as e:
            print(f"❌ Error en {pdf_file}: {str(e)[:80]}")
    
    print("=" * 60)
    print(f"📊 Total de chunks: {total_chunks}")
    return chunks, sources

def create_qdrant_index(chunks, sources, model_name="BAAI/bge-small-en-v1.5", collection_path="qdrant_db", device="cuda"):
    """
    Genera vectorstore Qdrant optimizado para GPU con bge-small-en-v1.5 (384d)
    
    ✅ bge-small-en-v1.5 en GPU: ~2 minutos para 4000 chunks
    ✅ bge-small-en-v1.5 en CPU: ~5 minutos para 4000 chunks
    """
    print(f"\n🧠 Cargando modelo: {model_name} en {device.upper()}...")
    start_time = time.time()
    
    # ✅ Cargar modelo en GPU (cuda) o CPU
    model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - start_time
    print(f"✅ Modelo cargado en {load_time:.1f}s")
    
    print(f"\n🔍 Generando embeddings con {model_name} ({device.upper()})...")
    print("   ⚡ Optimizado para GPU: batch_size=128")
    start_embed = time.time()
    
    # ✅ Optimizado para GPU: batch_size mayor
    batch_size = 128 if device == "cuda" else 32
    embeddings = model.encode(
        chunks, 
        show_progress_bar=True, 
        normalize_embeddings=True,
        batch_size=batch_size,
        device=device
    )
    
    embed_time = time.time() - start_embed
    print(f"✅ Embeddings generados: {embeddings.shape} en {embed_time/60:.1f} minutos")
    
    # Inicializar cliente Qdrant en modo local (disco)
    client = QdrantClient(path=collection_path)
    
    # Crear colección
    client.create_collection(
        collection_name="acreditacion",
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    
    # ✅ CORRECCIÓN CRÍTICA: Usar PointStruct (no diccionarios)
    print("\n💾 Guardando en Qdrant con PointStruct...")
    start_upsert = time.time()
    
    # Crear puntos con PointStruct
    points = []
    for i, (chunk, source, embedding) in enumerate(zip(chunks, sources, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # ✅ ID como string
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_id": i
                }
            )
        )
    
    # Upsert en lotes
    batch_size = 256
    total_points = len(points)
    for i in range(0, total_points, batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name="acreditacion",
            points=batch
        )
        progress = min(i + batch_size, total_points)
        pct = progress / total_points * 100
        print(f"  → {progress}/{total_points} puntos ({pct:.1f}%)")
    
    upsert_time = time.time() - start_upsert
    print(f"✅ Índice Qdrant guardado en '{collection_path}/' en {upsert_time:.1f}s")
    
    # ✅ Exportar chunks para BM25
    bm25_data_path = os.path.join(collection_path, "bm25_data.pkl")
    with open(bm25_data_path, "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)
    print(f"✅ Chunks exportados para BM25: {bm25_data_path}")
    
    total_time = time.time() - start_time
    print(f"\n📚 Modelo: {model_name} ({embeddings.shape[1]} dimensiones)")
    print(f"📄 Chunks indexados: {len(chunks)}")
    print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 GENERADOR DE VECTORSTORE QDRANT + BM25 CON BGE-SMALL (GPU)")
    print("=" * 70)
    print(f"\n⚙️  Configuración:")
    print(f"   • Modelo: BAAI/bge-small-en-v1.5 (384 dimensiones)")
    print(f"   • Device: {DEVICE.upper()}")
    print(f"   • Batch size: {128 if DEVICE == 'cuda' else 32}")
    print(f"\n⏱️  Tiempo estimado:")
    print(f"   • GPU (RTX 3060+): 2-3 minutos para 4000 chunks")
    print(f"   • CPU: 5-7 minutos para 4000 chunks")
    print("=" * 70)
    
    clean_qdrant_db("qdrant_db")
    chunks, sources = process_pdfs("pdfs")
    
    if chunks:
        create_qdrant_index(
            chunks, 
            sources, 
            model_name="BAAI/bge-small-en-v1.5",  # ✅ LIGERO Y RÁPIDO
            collection_path="qdrant_db",
            device=DEVICE
        )
        
        print("\n" + "=" * 70)
        print("✅ ¡VECTORSTORE CON BGE-SMALL CREADO EXITOSAMENTE!")
        print("=" * 70)
        print("\n📌 PRÓXIMOS PASOS:")
        print("1. Comprime la carpeta qdrant_db:")
        print("   Compress-Archive -Path 'qdrant_db' -DestinationPath 'qdrant_db.zip'")
        print("2. Verifica tamaño (< 50 MB):")
        print("   (Get-Item qdrant_db.zip).Length / 1MB")
        print("3. Sube qdrant_db.zip a GitHub")
        print("4. Usa app.py con RAG híbrido (BM25 + Qdrant + DeepSeek)")
        print("\n🎉 Calidad de embeddings: ⭐⭐⭐⭐ (bge-small 384d)")
        print("   ✅ 100% compatible con Streamlit Cloud (sin 'Killed')")
        print("=" * 70)
    else:
        print("\n❌ No se generaron chunks. Verifica tus PDFs en la carpeta 'pdfs/'.")