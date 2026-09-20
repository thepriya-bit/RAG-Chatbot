import streamlit as st

from rag_pipeline import answer_question


st.title("📚 RAG Chatbot")

st.write("Ask questions about the uploaded document.")

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)

    st.write("### Answer")
    st.write(answer)