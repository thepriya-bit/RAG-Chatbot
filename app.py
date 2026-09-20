import hashlib

import streamlit as st

from document_processor import load_pdf, chunk_text
from embeddings import create_embeddings
from vector_store import add_documents
from rag_pipeline import answer_question


st.title("📚 RAG Chatbot")

st.write("Upload PDFs and ask questions about them.")


# -----------------------------------
# 1. Initialize chat history
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# 2. Track uploaded documents
# -----------------------------------

if "document_hashes" not in st.session_state:
    st.session_state.document_hashes = set()


# -----------------------------------
# 3. PDF upload
# -----------------------------------

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        # Read PDF
        file_bytes = uploaded_file.getvalue()

        # Create unique hash
        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()


        # -----------------------------------
        # Process only new PDFs
        # -----------------------------------

        if file_hash not in st.session_state.document_hashes:

            # Save temporarily
            with open(
                "uploaded_document.pdf",
                "wb"
            ) as f:
                f.write(file_bytes)


            # Extract text
            text = load_pdf(
                "uploaded_document.pdf"
            )


            # Create chunks
            chunks = chunk_text(text)


            # Create embeddings
            embeddings = create_embeddings(
                chunks
            )


            # Store document in ChromaDB
            add_documents(
                chunks,
                embeddings,
                uploaded_file.name,
                file_hash
            )


            # Remember document
            st.session_state.document_hashes.add(
                file_hash
            )


            st.success(
                f"{uploaded_file.name} processed successfully!"
            )


    # -----------------------------------
    # 4. Display chat history
    # -----------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])


    # -----------------------------------
    # 5. Chat input
    # -----------------------------------

    question = st.chat_input(
        "Ask a question about the uploaded PDFs"
    )


    if question:

        with st.chat_message("user"):
            st.write(question)


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = answer_question(
                    question,
                    st.session_state.messages
                )

            st.write(answer)


        # Save conversation
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )