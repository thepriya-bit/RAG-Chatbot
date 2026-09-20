import streamlit as st

from document_processor import load_pdf, chunk_text
from embeddings import create_embeddings
from vector_store import add_documents, clear_collection
from rag_pipeline import answer_question


st.title("📚 RAG Chatbot")

st.write("Upload a PDF and ask questions about it.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track whether a document has been processed
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Process the PDF only once
    if not st.session_state.document_processed:

        # Save uploaded PDF
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        text = load_pdf("uploaded_document.pdf")

        # Create chunks
        chunks = chunk_text(text)

        # Create embeddings
        embeddings = create_embeddings(chunks)

        # Clear previous document
        clear_collection()

        # Store new document
        add_documents(chunks, embeddings)

        # Mark document as processed
        st.session_state.document_processed = True

        st.success("PDF processed successfully!")


    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Chat input
    question = st.chat_input(
        "Ask a question about the PDF"
    )


    if question:

        # Display user message
        with st.chat_message("user"):
            st.write(question)


        # Generate answer
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