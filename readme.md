# Ramayana RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers
questions using information retrieved from a Ramayana PDF.

## Features

- PDF document loading and text extraction
- Recursive text chunking with overlap
- Semantic embeddings using Hugging Face
- Vector indexing and similarity search using FAISS
- Answer generation using Groq-hosted LLMs
- Interactive chatbot interface using Streamlit
- Source passage display for retrieved context

## Tech Stack

- Python
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Groq
- Streamlit

## Architecture

PDF → Chunking → Embeddings → FAISS → Retrieval → Groq LLM

## Run Locally

1. Clone the repository.
2. Install dependencies.
3. Configure your Groq API key.
4. Run the Streamlit application.

See the project setup instructions for details.