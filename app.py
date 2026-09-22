import hashlib

import streamlit as st

from document_processor import load_pdf, chunk_text
from embeddings import create_embeddings
from vector_store import (
    add_documents,
    clear_collection,
    document_exists
)
from rag_pipeline import answer_question
st.title("📚 RAG Chatbot")
st.write("Upload PDFs and ask questions about them.")


    # --------------------------------
    # Session state
    # --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "document_hashes" not in st.session_state:
    st.session_state.document_hashes = set()


if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()


if "just_cleared" not in st.session_state:
    st.session_state.just_cleared = False


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


    # --------------------------------
    # Clear documents button
    # --------------------------------

if st.button("🗑️ Clear All Documents"):
    clear_collection()
    st.session_state.document_hashes = set()

    st.session_state.processed_hashes = set()

    st.session_state.messages = []

    st.session_state.uploader_key += 1

    st.session_state.just_cleared = True

    st.success(
        "All documents and chat history cleared."
    )
    st.rerun()


    # --------------------------------
    # PDF uploader
    # --------------------------------

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}"
)


    # --------------------------------
    # Process uploaded documents
    # --------------------------------

if uploaded_files and not st.session_state.just_cleared:

    for uploaded_file in uploaded_files:

            # --------------------------------
            # Calculate file hash
            # --------------------------------

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()


            # --------------------------------
            # Check if already uploaded
            # --------------------------------

        if file_hash in st.session_state.processed_hashes:
            continue


            # --------------------------------
            # Check ChromaDB
            # --------------------------------

        if document_exists(file_hash):

            st.info(
                f"⚠️ {uploaded_file.name} "
                "is already in the knowledge base."
            )

            st.session_state.document_hashes.add(
                file_hash
            )

            st.session_state.processed_hashes.add(
                file_hash
            )

            continue


            # --------------------------------
            # Save temporary PDF
            # --------------------------------

        with open(
            "uploaded_document.pdf",
            "wb"
        ) as f:

            f.write(file_bytes)


            # --------------------------------
            # Extract text
            # --------------------------------

        text = load_pdf(
            "uploaded_document.pdf"
        )


            # --------------------------------
            # Create chunks
            # --------------------------------
        chunks = chunk_text(text)
        print("FILE:", uploaded_file.name)
        print("EXTRACTED TEXT LENGTH:", len(text))
        print("NUMBER OF CHUNKS:", len(chunks))
        print("CHUNKS:", chunks[:2])
        if not chunks:
            st.error(
                f"❌ No text could be extracted from {uploaded_file.name}."
            )
            continue
        embeddings = create_embeddings(chunks)
        print("NUMBER OF EMBEDDINGS:", len(embeddings))
        if len(embeddings) == 0:
            st.error(
                f"❌ Failed to create embeddings for {uploaded_file.name}."
            )
            continue
        


            # --------------------------------
            # Add document to ChromaDB
            # --------------------------------

        added = add_documents(
            chunks,
            embeddings,
            uploaded_file.name,
            file_hash
        )


            # --------------------------------
            # Remember document
            # --------------------------------

        st.session_state.document_hashes.add(
            file_hash
        )

        st.session_state.processed_hashes.add(
            file_hash
        )


            # --------------------------------
            # Show result
            # --------------------------------

        if added:

            st.success(
                f"✅ {uploaded_file.name} "
                "processed successfully!"
            )

        else:

            st.info(
                f"⚠️ {uploaded_file.name} "
                "is already in the knowledge base."
            )


    # --------------------------------
    # Reset clear flag
    # --------------------------------

if st.session_state.just_cleared:

    st.session_state.just_cleared = False


    # --------------------------------
    # Display previous messages
    # --------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


    # --------------------------------
    # Chat input
    # --------------------------------

question = st.chat_input(
    "Ask a question about the uploaded PDFs"
)


if question:

        # --------------------------------
        # Display user question
        # --------------------------------

    with st.chat_message("user"):

        st.write(question)


        # --------------------------------
        # Generate answer
        # --------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = answer_question(
                question,
                st.session_state.messages
            )

        st.write(answer)


        # --------------------------------
        # Save conversation
        # --------------------------------

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