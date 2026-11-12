# Alzheimer's Research Chatbot — RAG + LLM

A Retrieval-Augmented Generation (RAG) chatbot that answers questions 
about Alzheimer's research using PubMed papers as its knowledge base.

## How it works

1. **Fetch** — Downloads Alzheimer's research abstracts from PubMed API
2. **Chunk** — Splits papers into manageable text chunks
3. **Embed** — Creates vector embeddings and stores them in ChromaDB
4. **Query** — Retrieves relevant chunks and passes them to an LLM
5. **Answer** — Returns an answer with source attribution

## Pipeline

| File | Purpose |
|------|---------|
| `1_fetch_papers.py` | Fetch papers from PubMed API |
| `2_chunk_papers.py` | Split papers into chunks |
| `3_create_vector_db.py` | Build ChromaDB vector store |
| `query_processor.py` | Retrieve + generate answers |
| `app.py` | Streamlit app interface |

## Data

- `raw_pubmed_papers.csv` — fetched paper metadata and abstracts
- `paper_chunks.csv` — chunked text for embedding
- `chroma_db/` — persisted vector database

## Stack

- Python (pandas, requests)
- ChromaDB (vector store)
- LLM API (OpenAI / compatible)
- Streamlit (interface)
- PubMed API (data source)

## How to run

```bash
pip install -r requirements.txt
python 1_fetch_papers.py
python 2_chunk_papers.py
python 3_create_vector_db.py
streamlit run app.py

## Attribution

Starter pipeline structure adapted from an open-source facial 
expression recognition project. Model training (28,000+ images), 
detection pipeline, and deployment implemented by Arphaxad Nguka Owange.