# PDF Question-Answering Chatbot (RAG)

Upload a PDF, retrieve semantically relevant chunks using Sentence-Transformers + FAISS, and answer questions from retrieved passages. If an OpenAI API key is configured, the app uses it to compose a grounded answer. Without a key, it provides an extractive answer from the most relevant passages.

## Setup
Python 3.10+ recommended. `sentence-transformers` downloads its embedding model on first run, so the first launch needs internet access.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Set `OPENAI_API_KEY` in `.env` only if you want LLM-generated answers. Never commit `.env` or API keys to GitHub.

## How it works
1. Extract text from each PDF page with pypdf.
2. Split text into overlapping chunks.
3. Encode chunks using `sentence-transformers/all-MiniLM-L6-v2`.
4. Index vectors in FAISS and retrieve top-k passages.
5. Generate an answer from those passages, or show a grounded extractive response.
6. Display source page numbers for transparency.

## Limitations
Scanned/image-only PDFs need OCR (not included). Answers can be incomplete if the PDF has poor text extraction or the relevant passage is not retrieved. Validate important answers against the source PDF.
