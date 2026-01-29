sample.pdf
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

agenticrag.py
🏨 Agentic Multi-PDF Question Answering System (LangGraph)

This project implements an agent-based Retrieval-Augmented Generation (RAG) system that allows users to ask questions over multiple PDF documents using a structured LangGraph workflow.

The agent intelligently detects intent, retrieves relevant document content, generates grounded answers, and evaluates response quality — all in an interactive command-line interface.

✨ Features

📄 Loads multiple PDF documents

📑 Page-wise PDF text extraction

🧠 Semantic search using OpenAI embeddings

🔁 Agentic workflow using LangGraph

🧪 Response evaluation with similarity scoring

💬 Interactive CLI chat experience

🔐 Secure API key handling via .env

🧠 Agent Workflow

The system follows a state-driven agent loop:

User Input
   ↓
Intent Detection
   ↓
Document Retrieval
   ↓
Answer Generation
   ↓
Answer Evaluation
   ↓
Decision (Retrieve Again or End)


Each stage is implemented as a LangGraph node, making the system modular and extensible.

🗂️ Project Structure
project/
│
├── main.py              # Complete agent workflow implementation
├── .env                 # Environment variables (API key)
├── README.md            # Documentation
├── *.pdf                # Input PDF documents

📄 Document Handling

PDFs are parsed using Unstructured

Text is grouped page-wise

Each chunk stores:

PDF filename

Page number

Extracted text

This allows precise, source-aware answers.

🔍 Retrieval Strategy

User query is converted to an embedding

All document chunks are embedded

Cosine similarity ranks relevance

Top-K relevant chunks are selected

Keyword search is used as a fallback if embeddings fail

If no relevant data is found, the agent responds safely without hallucinating.

🧪 Answer Evaluation

After generating a response, the agent:

Embeds the answer

Compares it with retrieved context

Produces a confidence score (0.0 – 1.0)

This score reflects how well the answer aligns with document content.

⚙️ Tech Stack

Python

LangGraph – agent orchestration

OpenAI API – embeddings & generation

Unstructured – PDF parsing

dotenv – environment variable management

Typing / StateGraph – structured agent state

▶️ How to Run
1️⃣ Install dependencies
pip install openai langgraph unstructured python-dotenv

2️⃣ Set environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here

3️⃣ Add PDF files

Place your PDFs in the project directory (or update paths in the code).

4️⃣ Run the application
python main.py

💬 Usage

Ask questions in natural language

The agent answers using only document knowledge

Confidence score is shown after each response

Type exit to quit

Example:

You: What are the refund rules for non-refundable tickets?
Agent: [Answer based on PDF content]
(Confidence: 0.81/1.00)

🔐 Security & Privacy

✅ No API keys hard-coded

✅ Uses environment variables

✅ No external browsing by default

✅ No internal reasoning exposed to users

Safe for:

College assignments

GitHub repositories

Project demos

🚀 Possible Enhancements

Web UI (Streamlit / FastAPI)

Persistent vector storage

Multi-language document support

Better chunking strategies

PDF upload support

📌 Disclaimer

This system answers questions only based on the provided PDF documents.
If the information is not present, the agent clearly states that it does not know.




