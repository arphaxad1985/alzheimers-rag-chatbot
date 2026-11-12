# query_processor.py
from sentence_transformers import SentenceTransformer
import chromadb
import openai
import os

# Load the same embedding model used for indexing
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the existing ChromaDB using the new PersistentClient
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = chroma_client.get_collection("pubmed_papers")

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_user_query(user_question, top_k=5):
    """
    Complete pipeline from user question to cited answer
    """
    print(f"👤 User: {user_question}")
    
    # STEP 4-5: Embed the user question
    print("🔄 Converting question to embedding...")
    question_embedding = model.encode([user_question]).tolist()
    
    # STEP 6: Semantic search in Vector DB
    print("🔍 Searching for relevant papers...")
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    
    # Extract retrieved chunks and metadata
    retrieved_chunks = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    
    print(f"✅ Found {len(retrieved_chunks)} relevant chunks")
    
    # STEP 7: Augmented Prompting
    context = "\n\n".join([
        f"[Source {i+1} - PMID: {meta['pmid']}]\n{chunk}" 
        for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_metadata))
    ])
    
    prompt = f"""You are a helpful medical research assistant. Answer the user's question based ONLY on the provided research paper excerpts. If the answer is not in the excerpts, say "I cannot find information about this in the available research."

CONTEXT FROM RESEARCH PAPERS:
{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the context provided
2. Cite the source PMID for each claim using [Source X]
3. If the context doesn't contain relevant information, say so honestly
4. Do not use any external knowledge or make assumptions

ANSWER:"""
    
    # STEP 8: LLM Generation
    print("🤖 Generating answer with LLM...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a precise medical research assistant that only answers based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Low temperature for factual answers
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    # STEP 9: Response & Citation
    return {
        'question': user_question,
        'answer': answer,
        'sources': [
            {
                'pmid': meta['pmid'],
                'chunk': chunk[:200] + "..."  # Preview
            }
            for chunk, meta in zip(retrieved_chunks, retrieved_metadata)
        ]
    }