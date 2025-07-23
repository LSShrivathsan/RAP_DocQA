# Document Q&A Chatbot

This is a Document Question Answering (DocQA) system built with Gradio and the Qwen LLMs. It enables users to upload documents (PDF, DOCX, TXT), semantically index them, and ask questions based on the content. The system uses RAG and embedding-based retrieval combined with a generative LLM to provide context-aware answers.

---

## Features

-  Embedding-based semantic search with Qwen embeddings  
-  Question answering with `Qwen/Qwen-7B-Chat`  
-  Upload and parse `.pdf`, `.docx`, and `.txt` files  
-  Vector search with FAISS  
-  Fast, local inference via PyTorch with GPU/CPU support  
-  Gradio UI for simple interactive testing  

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/LSShrivathsan/RAP_DocQA.git

cd RAP_DocQA

### 2. Create environment and install dependencies
python -m venv venv

source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt

### 3. Run the application
python app.py

Gradio will display a URL to access the UI in your browser.

## Architecture & Design Decisions

## System Components
- Component	        Technology
- Embedding Model	  Qwen/Qwen3-Embedding-0.6B
- LLM	              Qwen/Qwen-7B-Chat
- Index	            faiss.IndexFlatL2
- UI	              gradio

## Architectural Flow
Document Upload: Users upload files through Gradio UI

Text Extraction: Based on file type using PyPDF2, python-docx, or basic decoding

Chunking: Text split into 256 token chunks

Embedding: Each chunk embedded with Qwen Embedding model

Indexing: FAISS used to build a searchable vector store

### QA Flow:

User asks a question

Top 3 relevant chunks retrieved

Prompt built with context

Qwen-7B-Chat generates a final answer

## Chunking Strategy
Method: Naive chunking using 256 tokens per chunk

Rationale: Simple, fast, and compatible with Qwen embedding model

Future Improvement: Use overlap based chunking (e.g., sliding window)

## Retrieval Approach
Vector Search: FAISS with IndexFlatL2

Embedding: Qwen3-Embedding-0.6B (fast and high-quality)

Top-k: Retrieves top 3 chunks based on cosine similarity

Context Window: Selected chunks combined into a prompt for answering

## Hardware & Performance Observations
Embedding Model: Lightweight, suitable for CPU or GPU

Chat Model (Qwen-7B-Chat):

Requires GPU with 12 GB VRAM minimum 

Memory Usage:

RAM: ~2–3GB for embedding + FAISS index

VRAM: ~10–14GB for model in float16

Inference Speed : 2s - 15s  

## File Support
- Format	Parser
- .pdf	  PyPDF2
- .docx	  python-docx
- .txt	  UTF-8 decoding
