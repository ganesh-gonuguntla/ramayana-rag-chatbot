from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Ramayana RAG Chatbot",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Ramayana RAG Chatbot")

st.write(
    "Ask questions about the Ramayana and get answers "
    "grounded in multiple reference documents."
)


# --------------------------------------------------
# 2. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"


# --------------------------------------------------
# 3. Load all PDFs and build the FAISS vector store
# --------------------------------------------------

@st.cache_resource
def create_vector_store():

    # Find all PDF files inside the documents directory
    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found inside the documents folder."
        )

    all_documents = []

    # Load each PDF
    for pdf_path in pdf_files:

        print(f"Loading document: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        if not documents:
            print(f"Warning: No pages extracted from {pdf_path.name}")
            continue

        # Store the source filename in the metadata
        for document in documents:
            document.metadata["source"] = pdf_path.name

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError(
            "No extractable pages were found in the PDF documents."
        )

    print(f"Total pages loaded: {len(all_documents)}")

    # Split the combined documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    if not chunks:
        raise ValueError(
            "No text chunks were extracted from the PDF documents."
        )

    print(f"Total chunks created: {len(chunks)}")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build one FAISS index containing chunks from every PDF
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print("FAISS vector store created successfully.")

    return vector_store


# --------------------------------------------------
# 4. Initialize the Groq LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    api_key = st.secrets["GROQ_API_KEY"]

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0
    )


# --------------------------------------------------
# 5. Create the RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant specializing in the Ramayana.

Answer the user's question using the retrieved context
from the available reference documents.

Instructions:
1. Ground your answer in the provided context.
2. Do not invent facts or events unsupported by the context.
3. If the answer cannot be found in the context, clearly
   state that you could not find sufficient information
   in the available documents.
4. If the documents present different versions or
   interpretations of an event, explain the differences
   when supported by the retrieved context.
5. Do not assume that all Ramayana traditions are identical.
6. Give a clear, concise, and understandable answer.
7. Do not claim that a source says something unless
   the retrieved context supports that claim.

Retrieved context:
{context}

User question:
{question}

Answer:
""")


# --------------------------------------------------
# 6. Initialize resources
# --------------------------------------------------

try:

    with st.spinner(
        "Loading documents and preparing the knowledge base..."
    ):

        vector_store = create_vector_store()
        llm = create_llm()

except Exception:

    st.error(
        "Could not initialize the chatbot. "
        "Please check your PDF documents, dependencies, "
        "and Streamlit secrets."
    )

    st.stop()


# --------------------------------------------------
# 7. Create the retriever and RAG chain
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

rag_chain = prompt | llm


# --------------------------------------------------
# 8. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 9. Handle user input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the Ramayana..."
)

if question:

    # Save and display the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve context and generate the answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the reference documents..."
        ):

            try:

                # Retrieve relevant chunks from both PDFs
                retrieved_docs = retriever.invoke(question)

                # Combine retrieved passages
                context = "\n\n".join(
                    (
                        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                        f"{doc.page_content}"
                    )
                    for doc in retrieved_docs
                )

                # Generate the answer
                response = rag_chain.invoke({
                    "context": context,
                    "question": question
                })

                answer = response.content

                st.markdown(answer)

                # Display the retrieved source passages
                with st.expander("View source passages"):

                    if not retrieved_docs:
                        st.write(
                            "No relevant passages were retrieved."
                        )

                    for index, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        source = doc.metadata.get(
                            "source",
                            "Unknown document"
                        )

                        page = doc.metadata.get(
                            "page",
                            None
                        )

                        # PyPDFLoader typically uses zero-based pages
                        page_label = (
                            page + 1
                            if isinstance(page, int)
                            else "Unknown"
                        )

                        st.markdown(
                            f"**Passage {index}**"
                        )

                        st.caption(
                            f"Document: {source} | Page: {page_label}"
                        )

                        st.write(doc.page_content)

                        st.divider()

                # Save the assistant's response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception:

                st.error(
                    "Sorry, something went wrong while "
                    "retrieving information or generating "
                    "the answer. Please try again."
                )

