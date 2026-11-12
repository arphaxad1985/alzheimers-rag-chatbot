# create_vector_db.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

# Initialize embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
print("✅ Model loaded")

# Initialize ChromaDB client
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(allow_reset=True)  # optional
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="pubmed_papers",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Load chunks
chunks_df = pd.read_csv("paper_chunks.csv")
print(f"📊 Processing {len(chunks_df)} chunks")

# Process in batches to avoid memory issues
batch_size = 100
for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    
    # Generate embeddings
    embeddings = model.encode(batch['chunk_text'].tolist()).tolist()
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=batch['chunk_text'].tolist(),
        metadatas=[{'pmid': row['pmid'], 'chunk_id': row['chunk_id']} 
                   for _, row in batch.iterrows()],
        ids=batch['chunk_id'].tolist()
    )
    
    print(f"✅ Added batch {i//batch_size + 1}/{(len(chunks_df)//batch_size)+1}")

print("🎉 Vector database created successfully!")