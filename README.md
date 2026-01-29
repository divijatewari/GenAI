This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions about the contents of a PDF document. The system retrieves relevant information from the document and generates accurate answers using a local Large Language Model (LLM).

All sensitive information such as file names, API keys, internal traces, and reasoning steps are intentionally excluded to ensure privacy and safe sharing.

🚀 Features

📑 Extracts text and tables from PDF documents

✂️ Splits content into overlapping chunks for better retrieval

🔗 Generates semantic embeddings using Sentence Transformers

⚡ Fast similarity search with FAISS

🤖 Local LLM-based answer generation

🔐 No exposure of private data, tool traces, or internal logs

🧠 How It Works

PDF Extraction
Text and tables are extracted page by page from the input PDF.

Chunking
Extracted content is split into overlapping chunks to preserve context.

Embedding
Each chunk is converted into a vector representation using a sentence embedding model.

Indexing
FAISS is used to index embeddings for efficient similarity search.

Retrieval
For a user query, the most relevant chunks are retrieved from the index.

Answer Generation
A local LLM generates the final answer using only the retrieved context.

📦 Tech Stack

Python

pdfplumber – PDF text & table extraction

SentenceTransformers – Semantic embeddings

FAISS – Vector similarity search

Local LLM API (e.g., Ollama)

📁 Project Structure
project/
│
├── main.py              # Core RAG pipeline
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation

▶️ How to Run

Install dependencies

pip install -r requirements.txt


Start your local LLM service
(Ensure a compatible local LLM is running.)

Run the program

python main.py


Provide input

Enter the path to a PDF file



🎯 Use Cases

Academic PDF analysis

Policy or document Q&A

Knowledge extraction from reports

Offline or private document querying

Ask questions about its content

Type exit to quit
