# app.py - Complete Alzheimer's Research Chatbot
import streamlit as st
from query_processor import process_user_query
import pandas as pd

st.set_page_config(page_title="Alzheimer's Research Assistant", layout="wide")

st.title("🧠 Alzheimer's Research Assistant")
st.markdown("""
This chatbot answers questions based on real PubMed research papers about Alzheimer's disease.
**Features:**
- 📚 Live PubMed data
- 🔍 Semantic search (finds by meaning, not just keywords)
- ✅ Cited, traceable answers
- 🤔 Honest "I don't know" responses
""")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**PMID: {source['pmid']}**")
                    st.caption(source['chunk'])

# Chat input
if prompt := st.chat_input("Ask about Alzheimer's research..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching PubMed papers..."):
            result = process_user_query(prompt)
        
        st.markdown(result['answer'])
        
        # Show sources
        with st.expander("📚 View Sources"):
            for source in result['sources']:
                st.markdown(f"**PMID: {source['pmid']}**")
                st.caption(source['chunk'])
                st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/)")
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result['answer'],
            "sources": result['sources']
        })

# Sidebar with stats
with st.sidebar:
    st.header("📊 Database Stats")
    try:
        chunks_df = pd.read_csv("paper_chunks.csv")
        papers_df = pd.read_csv("raw_pubmed_papers.csv")
        st.metric("Papers Indexed", len(papers_df))
        st.metric("Text Chunks", len(chunks_df))
    except:
        st.warning("Run the offline preparation first!")
    
    st.header("🔧 How It Works")
    st.markdown("""
    1. **Offline**: Downloads PubMed papers
    2. **Chunks**: Splits into pieces
    3. **Embeddings**: Converts to vectors
    4. **Vector DB**: Stores in Chroma
    5. **Query**: Your question embedded
    6. **Search**: Finds similar meaning
    7. **Generate**: LLM answers with citations
    """)