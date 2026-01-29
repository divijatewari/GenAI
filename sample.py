import re
import faiss
import pdfplumber
import numpy as np
import requests
from textwrap import dedent
from sentence_transformers import SentenceTransformer


# ---- PDF Extraction ----
def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()

            table_data = ""
            for table in tables:
                for row in table:
                    table_data += " | ".join(str(cell) if cell else "" for cell in row) + "\n"

            combined = f"[Page {i+1}]\n{text}\n{table_data}"
            pages.append(combined)

    return pages


# ---- Chunking ----
def chunk_text(pages, chunk_size=800, overlap=100):
    chunks = []
    for page in pages:
        for i in range(0, len(page), chunk_size - overlap):
            chunks.append(page[i:i + chunk_size])
    return chunks


# ---- Embedding ----
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks, show_progress_bar=False)
    return vectors, model


# ---- FAISS Index ----
def create_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype=np.float32))
    return index


# ---- Search ----
def search_chunks(query, model, index, chunks, k=4):
    q_vec = model.encode([query])
    _, indices = index.search(np.array(q_vec, dtype=np.float32), k)
    return [chunks[i] for i in indices[0]]


# ---- LLM Call (Local / Abstracted) ----
def call_llm(prompt, model="llama3"):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return res.json().get("response", "")
    except:
        return ""


# ---- Agent Controller (No Traces) ----
def rag_agent(query, model, index, chunks, pages):
    context = "\n\n".join(search_chunks(query, model, index, chunks))

    prompt = dedent(f"""
        Answer the question using ONLY the context below.

        CONTEXT:
        {context}

        QUESTION:
        {query}
    """)

    return call_llm(prompt).strip()


# ---- MAIN ----
if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip()

    pages = extract_pdf(pdf_path)
    chunks = chunk_text(pages)
    vectors, embedder = embed_chunks(chunks)
    index = create_index(vectors)

    while True:
        query = input("\nAsk a question (or type exit): ").strip()
        if query.lower() == "exit":
            break

        answer = rag_agent(query, embedder, index, chunks, pages)
        print("\nAnswer:\n", answer)
