# fetch_papers.py
from Bio import Entrez
import time
import pandas as pd

# Configure your email (required by NCBI)
Entrez.email = "arphaxadnguka@gmail.com"

def search_pubmed(query, max_results=100):
    """
    Search PubMed for papers matching a query
    """
    print(f"🔍 Searching PubMed for: {query}")
    
    # Search for papers
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    pmids = record["IdList"]
    print(f"✅ Found {len(pmids)} papers")
    return pmids

def fetch_abstracts(pmids):
    """
    Fetch full abstracts for given PMIDs
    """
    papers = []
    
    for i, pmid in enumerate(pmids):
        if i % 10 == 0:
            print(f"📄 Fetching paper {i+1}/{len(pmids)}")
        
        try:
            # Fetch paper details
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract = handle.read()
            handle.close()
            
            # Fetch metadata (title, authors, journal)
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            medline = handle.read()
            handle.close()
            
            # Simple parsing (in production, use XML parsing)
            papers.append({
                'pmid': pmid,
                'abstract': abstract,
                'medline': medline
            })
            
            # Be nice to NCBI servers
            time.sleep(0.34)  # Max 3 requests per second
            
        except Exception as e:
            print(f"❌ Error fetching {pmid}: {e}")
    
    return papers

# Run the fetch
pmids = search_pubmed("Alzheimer's disease treatment", max_results=50)
papers = fetch_abstracts(pmids)

# Save raw data
df = pd.DataFrame(papers)
df.to_csv("raw_pubmed_papers.csv", index=False)
print("✅ Saved raw papers to raw_pubmed_papers.csv")