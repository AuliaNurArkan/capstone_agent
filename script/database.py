from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import pandas as pd
from dotenv import load_dotenv
import os
import re
import logging
import tiktoken

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Clients
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300 
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Tiktoken Encoder (untuk evaluasi statistik)
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# --- HELPER FUNCTIONS ---

def clean_html_regex(html_content):
    if not html_content or pd.isna(html_content): return ""
    text = str(html_content)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def identify_sections(text):
    """Segmentasi sesuai aturan Section Segmentation Rules."""
    headers = [
        "SUMMARY", "OBJECTIVE", "PROFILE",
        "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT",
        "PROJECTS", "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
        "EDUCATION", "CERTIFICATIONS", "ACHIEVEMENTS", "CONTACT"
    ]
    pattern = r'(?i)^\s*(' + '|'.join(headers) + r')\s*$'
    parts = re.split(pattern, text, flags=re.MULTILINE)
    
    sections = {}
    current_sec = "GENERAL"
    if parts[0].strip(): sections[current_sec] = parts[0].strip()
    for i in range(1, len(parts), 2):
        header = parts[i].strip().upper()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        sections[header] = content
    return sections

def get_splitter(section_name):
    """Aturan Chunking sesuai Special Rules by Section."""
    if "SKILLS" in section_name:
        # Rule: 80-150 tokens
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    elif "EXPERIENCE" in section_name or "EMPLOYMENT" in section_name:
        # Rule: 120-280 tokens
        return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    else:
        # General: 220-350 tokens
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# --- STATISTIK & EVALUASI ---

def calculate_advanced_stats(orig_chars, documents):
    """Menampilkan statistik lengkap sebelum upload."""
    if not documents: return
    
    chunk_tokens = [len(encoder.encode(doc.page_content)) for doc in documents]
    avg_tokens = sum(chunk_tokens) / len(chunk_tokens)
    avg_orig = sum(orig_chars) / len(orig_chars)
    
    print("\n" + "="*60)
    print("📊 ANALISIS KUALITAS CHUNKING (TIKTOKEN & CHARS)")
    print("="*60)
    print(f"Total Resume Diproses    : {len(orig_chars)}")
    print(f"Rata-rata Char/Resume    : {avg_orig:.2f} chars")
    print(f"Total Chunks Dihasilkan  : {len(documents)}")
    print("-" * 60)
    print(f"Rata-rata Token/Chunk    : {avg_tokens:.2f} tokens")
    print(f"Token Terbanyak          : {max(chunk_tokens)} tokens")
    print(f"Token Tersedikit         : {min(chunk_tokens)} tokens")
    print("-" * 60)
    
    # Rekomendasi
    print("💡 ANALISIS UKURAN (CHUNK SIZE RECOMMENDATION):")
    if avg_tokens > 350:
        print("   ⚠️ STATUS: TERLALU BESAR. Potensi kehilangan konteks spesifik.")
    elif avg_tokens < 100:
        print("   ⚠️ STATUS: TERLALU KECIL. Konteks mungkin terlalu terfragmentasi.")
    else:
        print("   ✅ STATUS: IDEAL. Ukuran sudah sesuai dengan standar RAG (200-350 tokens).")
    print("="*60 + "\n")

# --- CORE FUNCTION ---

def load_data(excel_path: str, limit=None) -> list[Document]:
    logging.info(f"Loading data from {excel_path}...")
    df = pd.read_excel(excel_path)
    
    if limit:
        df = df.head(limit)

    # Rule: Maksimal 25 item per kategori
    df = df.groupby('Category').head(25).reset_index(drop=True)
    total_resumes = len(df)
    
    documents = []
    original_char_counts = [] 
    
    for idx, row in df.iterrows():
        resume_id = row.get('ID', idx)
        category = str(row.get('Category', 'Unknown'))
        
        # Rule: Source Selection
        content_str = str(row.get('Resume_str', ''))
        source_field = "Resume_str"
        if not content_str.strip() or content_str == 'nan':
            content_str = clean_html_regex(row.get('Resume_html', ''))
            source_field = "Resume_html"
        
        if not content_str.strip(): continue

        # Simpan hitungan karakter asli
        original_char_counts.append(len(content_str))

        # Rule: Section Segmentation
        sections = identify_sections(content_str)
        chunk_index = 0

        for sec_name, sec_text in sections.items():
            if not sec_text.strip(): continue
            
            splitter = get_splitter(sec_name)
            chunks = splitter.split_text(sec_text)

            for text_chunk in chunks:
                # Rule: Output Format (Text Prepend)
                final_text = f"SECTION: {sec_name}\n{text_chunk}"
                
                # Rule: Deterministic chunk_id
                c_id = f"resume:{resume_id}|sec:{sec_name}|idx:{chunk_index}"
                
                # Metadata Mapper
                metadata = {
                    "doc_type": "resume",
                    "resume_id": int(resume_id) if str(resume_id).isdigit() else str(resume_id),
                    "category": category,
                    "chunk_id": c_id,
                    "chunk_index": chunk_index,
                    "section": sec_name,
                    "source_field": source_field,
                    "language": "en",
                    "job_title": None, # Akan diisi model ekstraksi nantinya
                    "company": None,
                    "date_range": None
                }
                
                documents.append(Document(page_content=final_text, metadata=metadata))
                chunk_index += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_resumes:
            logging.info(f"Chunking Progress: {idx+1}/{total_resumes} resumes processed.")

    # TAMPILKAN STATISTIK SETELAH SELESAI CHUNKING
    calculate_advanced_stats(original_char_counts, documents)
    
    return documents

def main():
    # 1. Chunking & Stats
    documents = load_data('dataset/dataset.xlsx', limit=None)

    # 2. Upload (Berjalan setelah stats muncul)
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]

    if collection_name not in collection_names:
        logging.info(f"Creating collection {collection_name}...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    
    logging.info(f"Starting upload of {len(documents)} chunks to Qdrant...")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        logging.info(f"🚀 Upload Progress: {min(i + batch_size, len(documents))}/{len(documents)} chunks.")

    logging.info("ALL PROCESS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()