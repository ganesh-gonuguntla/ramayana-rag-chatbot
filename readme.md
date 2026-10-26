# Ramayana RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) application that lets you ask questions about the Ramayana and receive answers grounded in reference PDFs. The app loads the source documents, splits them into meaningful chunks, converts them to vector embeddings, retrieves the most relevant passages, and uses a Groq-hosted language model to generate evidence-based answers.

## Why this project matters

Traditional general-purpose language models can answer in a conversational way, but they may not reliably stay grounded in a specific source text. This project addresses that by combining:

- PDF ingestion from authoritative Ramayana reference documents
- Chunk-based retrieval for focused context selection
- Semantic search using embeddings and FAISS
- LLM answer generation with explicit source grounding
- A user-friendly chat interface built with Streamlit

This makes it useful for exploring stories, characters, events, and themes from the Ramayana in a controlled, source-aware way.

## Features

- Loads one or more PDF documents from the project data folder
- Extracts text from PDFs using PyPDF
- Splits content into overlapping chunks for better retrieval fidelity
- Uses Hugging Face embeddings to convert text into semantic vectors
- Stores vectors in FAISS for fast similarity search
- Retrieves the top relevant passages for a user question
- Generates grounded answers using Groq's LLM
- Shows the retrieved source passages in the UI for transparency
- Provides a simple chat experience through Streamlit

## Tech stack

- Python 3.10+
- Streamlit
- LangChain
- LangChain Community
- Hugging Face Sentence Transformers
- FAISS (CPU)
- Groq
- PyPDF

## Architecture

The application follows a standard RAG pipeline:

PDF documents → text extraction → chunking → embeddings → FAISS vector store → retrieval → Groq LLM → final answer

## Project structure

```text
Ramayana RAG/
├── app.py                  # Streamlit app and RAG logic
├── requirements.txt        # Python dependencies
├── Documents/              # Ramayana PDF files
├── .gitignore              # Git ignore rules
├── readme.md               # Project documentation
└── .streamlit/             # Optional local secrets folder
```

## Prerequisites

Before you run the app, make sure you have:

- Python 3.10 or newer
- A virtual environment tool such as `venv` or `conda`
- A Groq API key
- At least one Ramayana PDF in the `Documents/` folder

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ganesh-gonuguntla/ramayana-rag-chatbot.git
cd ramayana-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your PDFs

Place your Ramayana reference PDFs inside the `Documents/` folder. The app automatically loads all PDF files in that directory.

### 5. Configure the Groq API key

Create a `.streamlit/secrets.toml` file in the project root with the following content:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

If you do not already have a Groq API key, sign up at Groq and generate one from your account dashboard.

## Run the app

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## How it works

1. The app scans the `Documents/` directory for all `.pdf` files.
2. Each PDF is loaded and converted into page-level document chunks.
3. The text is split into overlapping chunks to preserve context around important passages.
4. Each chunk is embedded using a Hugging Face sentence transformer model.
5. The embeddings are stored in a FAISS vector database.
6. When the user asks a question, the system retrieves the most semantically relevant chunks.
7. The retrieved passages are injected into a prompt for the Groq LLM.
8. The model answers using the provided context, with a strong instruction to avoid unsupported claims.

## Example questions

- Who is Rama in the Ramayana?
- What is the significance of the exile of Rama?
- How does the Ramayana describe Sita's role?
- What are the major differences between the versions described in the source documents?
- Who are the key allies in Rama's journey?

## Expected behavior

The assistant is designed to be grounded in the loaded PDFs. It should:

- answer using retrieved context when enough evidence exists
- clearly state when the information is not present in the documents
- explain differing interpretations only when supported by the source text
- avoid making up unsupported details

## Troubleshooting

### No PDF files found

Make sure your source PDFs are inside the `Documents/` directory and that the file extensions are `.pdf`.

### Streamlit error on startup

Check that:

- dependencies were installed successfully
- the `.streamlit/secrets.toml` file exists
- the `GROQ_API_KEY` is valid and not empty

### Weak or incorrect answers

This usually happens when:

- the PDF files are incomplete or low quality
- too few relevant passages are retrieved
- the question is broader than the source material allows

Improve the model by adding richer source documents or using better chunking settings.

## Notes

- The app currently uses the `openai/gpt-oss-120b` model via Groq.
- Retrieval is based on chunk similarity, so answer quality depends heavily on the number and quality of the PDFs loaded.
- This is a research/demo-friendly implementation rather than a production-grade multi-user system.

## License

This project is provided as-is for educational and personal use. If you plan to distribute or commercialize it, confirm the applicable license terms for the source documents and models you use.

## Contributing

Contributions are welcome. If you want to improve the app:

- add better document handling
- improve retrieval quality
- support additional language models
- add evaluation and benchmarking for answers
- build a richer frontend or API layer

## Acknowledgements

- LangChain for the RAG workflow and abstractions
- Hugging Face for sentence-transformer embeddings
- FAISS for vector search
- Groq for hosted inference
- Streamlit for the interactive frontend

## Summary

This project demonstrates a practical RAG pipeline for a domain-specific knowledge base: the Ramayana. It combines semantic search, grounded generation, and an interactive interface to create a lightweight but effective AI-powered reading assistant.
