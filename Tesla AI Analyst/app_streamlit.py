import os
import pandas as pd
import yfinance as yf
import joblib

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Tesla_Annual_Report.pdf"
MODEL_PATH = BASE_DIR / "tsla_revenue_model.pkl"

# RAG: load & chunk PDF

@st.cache_resource
def load_pdf_text(path: str):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

@st.cache_resource
def build_rag_index():
    full_text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return chunks, embed_model, index

def retrieve_chunks(query: str, chunks, embed_model, index, k: int = 5):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    indices = indices[0]
    return [chunks[i] for i in indices]

# Local FLAN-T5-small model

@st.cache_resource
def load_qa_model():
    qa_model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)
    return tokenizer, model

def rag_answer_local(question: str, chunks, embed_model, index, tokenizer, qa_model, k: int = 5) -> str:
    retrieved = retrieve_chunks(question, chunks, embed_model, index, k=k)
    context = "\n\n---\n\n".join(retrieved)
    prompt = f"""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = qa_model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# ML: revenue forecast

def predict_next_year_revenue():
    artefact = joblib.load(MODEL_PATH)
    model = artefact["model"]
    feature_cols = artefact["feature_cols"]

    tsla = yf.Ticker("TSLA")
    fin = tsla.financials.T
    fin.index = fin.index.year

    df_latest = fin[feature_cols].copy()
    df_latest = df_latest.reset_index().rename(columns={"index": "year"})
    df_latest = df_latest.sort_values("year")

    latest_row = df_latest.iloc[-1]
    latest_year = int(latest_row["year"])
    X_latest = latest_row[feature_cols].to_frame().T

    pred = model.predict(X_latest)[0]
    return latest_year, pred

# Streamlit UI

def main():
    st.set_page_config(page_title="Tesla AI Analyst", layout="wide")
    st.title("🚗 Tesla AI Analyst: RAG + Revenue Forecast")

    st.write(
        "This app combines a Retrieval-Augmented Generation (RAG) chatbot over "
        "Tesla's annual report with a simple machine learning model that forecasts "
        "next year's total revenue based on historical financials. "
        "All models run locally with no external API keys."
    )

    chunks, embed_model, index = build_rag_index()
    tokenizer, qa_model = load_qa_model()

    tab1, tab2 = st.tabs(["💬 Chat with the Report", "📈 Revenue Forecast"])

    with tab1:
        st.subheader("Ask Questions about Tesla's Annual Report")
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the main risk factors Tesla highlights?"
        )
        if st.button("Ask", key="ask_rag") and question.strip():
            with st.spinner("Reading the report and generating answer..."):
                answer = rag_answer_local(
                    question, chunks, embed_model, index, tokenizer, qa_model, k=5
                )
            st.markdown("**Answer:**")
            st.write(answer)

    with tab2:
        st.subheader("Forecast Next Year's Revenue")
        if st.button("Run Forecast", key="forecast"):
            with st.spinner("Fetching financials and running model..."):
                latest_year, pred_revenue = predict_next_year_revenue()
            next_year = latest_year + 1
            st.markdown(f"**Latest year in data:** {latest_year}")
            st.markdown(f"**Predicted total revenue for {next_year}:** `{pred_revenue:,.0f} USD`")

if __name__ == "__main__":
    main()
