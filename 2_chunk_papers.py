# chunk_papers.py
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_abstract(text):
    """
    Basic cleaning of abstract text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common PubMed formatting
    text = text.replace('BACKGROUND:', '').replace('METHODS:', '').replace('RESULTS:', '').replace('CONCLUSIONS:', '')
    return text

def chunk_abstracts(papers_df, chunk_size=500, chunk_overlap=50):
    """
    Split abstracts into smaller chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    for idx, row in papers_df.iterrows():
        cleaned_text = clean_abstract(row['abstract'])
        
        # Split into chunks
        paper_chunks = text_splitter.split_text(cleaned_text)
        
        # Create chunks with unique IDs
        for chunk_idx, chunk_text in enumerate(paper_chunks):
            chunks.append({
                'chunk_id': f"{row['pmid']}_{chunk_idx}",
                'pmid': row['pmid'],
                'chunk_text': chunk_text,
                'chunk_index': chunk_idx
            })
    
    # Create DataFrame
    chunks_df = pd.DataFrame(chunks)
    return chunks_df

# Load papers
papers_df = pd.read_csv("raw_pubmed_papers.csv")
chunks_df = chunk_abstracts(papers_df)

# Save chunks
chunks_df.to_csv("paper_chunks.csv", index=False)
print(f"✅ Created {len(chunks_df)} chunks from {len(papers_df)} papers")