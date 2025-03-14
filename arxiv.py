import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Set up Streamlit page
st.set_page_config(page_title="arXiv Research Chatbot", page_icon="📚", layout="wide")

# Load sentence transformer for semantic search
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_model()

# Load summarization model
@st.cache_resource
def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

summarizer_tokenizer, summarizer_model = load_summarization_model()

# Load full arXiv dataset
@st.cache_data
def load_data():
    file_path = "c://Users//swathiga//Downloads//archive (14)//arxiv-metadata-oai-snapshot.json"  # Correct dataset file
    data = []
    
    # Read JSON file line by line to handle large datasets
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")
                return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip().str.lower()
    
    # Ensure required columns exist
    required_columns = {'title', 'abstract', 'categories', 'authors', 'id'}
    if not required_columns.issubset(df.columns):
        st.error("Error: Expected columns missing in the dataset!")
        return pd.DataFrame()
    
    # Filter for Computer Science papers
    df = df[df['categories'].str.contains('cs.', regex=True, na=False)]
    df['combined_text'] = df['title'] + " " + df['abstract']
    return df

df = load_data()

# Generate embeddings for papers
@st.cache_data
def generate_embeddings():
    if df.empty:
        return np.array([])  # Return empty array if no data is loaded
    return np.array([sentence_model.encode(text) for text in df['combined_text']])

embeddings = generate_embeddings()

# Function to search for relevant papers
def search_papers(query, top_n=5):
    if df.empty or embeddings.size == 0:
        st.error("No data available to search. Please check the dataset.")
        return pd.DataFrame(), []
    query_embedding = sentence_model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices], similarities[top_indices]

# Summarization function
def summarize_text(text, max_length=150):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=max_length, min_length=50, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("arXiv Research Chatbot")
st.subheader("Explore Scientific Papers in Computer Science")

query = st.text_input("Ask a research question:")
if query:
    with st.spinner("Searching for relevant papers..."):
        top_papers, similarities = search_papers(query)
    
    st.subheader("Relevant Papers")
    if not top_papers.empty:
        for i, paper in enumerate(top_papers.iterrows()):
            index, paper_data = paper
            with st.expander(f"{i+1}. {paper_data['title']}"):
                st.write(f"**Authors:** {paper_data['authors']}")
                st.write(f"**Categories:** {paper_data['categories']}")
                st.write(f"**Abstract:** {paper_data['abstract'][:500]}...")
                st.write(f"**Summary:** {summarize_text(paper_data['abstract'])}")
                st.write(f"[Read More](https://arxiv.org/abs/{paper_data['id']})")
    else:
        st.write("No relevant papers found.")

st.write("### Search for Computer Science papers and explore cutting-edge research!")
