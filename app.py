import os
import re
from pathlib import Path
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF Question-Answering Chatbot")
st.write("Ask questions about your PDF. Answers are grounded in retrieved document passages.")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_pages(uploaded_files):
    docs = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    docs.append({"source": file.name, "page": page_num, "text": text})
        except Exception as exc:
            st.warning(f"Could not read {file.name}: {exc}")
    return docs

def chunk_pages(pages, chunk_words=180, overlap=35):
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + chunk_words])
            if part:
                chunks.append({"source": page["source"], "page": page["page"], "text": part})
            if start + chunk_words >= len(words):
                break
    return chunks

def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve(question, chunks, index, embedder, top_k):
    qv = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(np.asarray(qv, dtype="float32"), min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            result = dict(chunks[int(idx)])
            result["score"] = float(score)
            results.append(result)
    return results

def make_answer(question, passages):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[Source: {p['source']}, page {p['page']}]\n{p['text']}" for p in passages
    )
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": (
                        "Answer only using the supplied PDF excerpts. If the excerpts do not contain "
                        "the answer, say that the uploaded document does not provide enough information. "
                        "Do not invent facts. Cite source filenames and page numbers in your answer."
                    )},
                    {"role": "user", "content": f"Question: {question}\n\nPDF excerpts:\n{context}"}
                ],
            )
            return response.choices[0].message.content, "LLM answer grounded in retrieved passages"
        except Exception as exc:
            st.warning(f"LLM request failed; using extractive fallback. Details: {exc}")
    # Fallback: show relevant retrieved excerpts rather than hallucinating an unsupported answer.
    excerpts = []
    for p in passages[:3]:
        excerpts.append(f"**{p['source']} — page {p['page']}**\n\n> {p['text']}")
    return "I could not generate a synthesized LLM answer (no API key configured or API request failed). Here are the most relevant retrieved excerpts:\n\n" + "\n\n".join(excerpts), "Extractive retrieval fallback"

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=4)
    st.caption("Embeddings run locally after the model is downloaded.")
    st.caption("Optional LLM mode requires an OpenAI API key in a local .env file.")

uploads = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
if uploads:
    if st.button("Process PDFs", type="primary"):
        with st.spinner("Extracting text and building the semantic search index..."):
            pages = extract_pages(uploads)
            chunks = chunk_pages(pages)
            if not chunks:
                st.error("No extractable text found. This may be a scanned PDF; OCR is not included.")
                st.stop()
            embedder = load_embedder()
            index = build_index(chunks, embedder)
            st.session_state["pdfqa"] = {"chunks": chunks, "index": index, "sources": [f.name for f in uploads]}
            st.success(f"Processed {len(uploads)} PDF(s), {len(pages)} text pages, and {len(chunks)} chunks.")
if "pdfqa" in st.session_state:
    state = st.session_state["pdfqa"]
    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="What are the main conclusions in this document?")
    if st.button("Get answer") and question.strip():
        with st.spinner("Searching document passages..."):
            embedder = load_embedder()
            passages = retrieve(question.strip(), state["chunks"], state["index"], embedder, top_k)
            answer, mode = make_answer(question.strip(), passages)
        st.markdown("### Answer")
        st.markdown(answer)
        st.caption(f"Mode: {mode}")
        with st.expander("Retrieved source passages"):
            for p in passages:
                st.markdown(f"**{p['source']} — page {p['page']}** · similarity {p['score']:.3f}")
                st.write(p["text"])
                st.divider()
else:
    st.info("Upload PDFs and click **Process PDFs** to begin.")
