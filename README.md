# RAG-Chatbot
This is my 1st Retrieval Augmented Generation project. So for the beginner level i'm making it naive.

a description-
# 📚 RAG Chatbot

A simple document-based Retrieval-Augmented Generation (RAG) chatbot built with Python.

The application allows users to upload PDF documents and ask questions about their contents. The system retrieves relevant document chunks and provides them to an LLM as context before generating an answer.

## 🚀 Features

- Upload one or multiple PDF documents
- Extract text from PDFs
- Structure-aware document chunking
- Generate vector embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Semantic similarity search
- Gemini-powered question answering
- Multi-document support
- Conversation history in Streamlit
- Duplicate document detection
- Clear document database and chat history
- Basic API error handling
- Grounded responses to reduce hallucination

## 🏗️ Architecture

```text
                PDF Documents
                     │
                     ▼
               PDF Extraction
                     │
                     ▼
                 Chunking
                     │
                     ▼
              Text Embeddings
                     │
                     ▼
                 ChromaDB
              Vector Database
                     │
                     │
User Question ───────┘
      │
      ▼
Question Embedding
      │
      ▼
Similarity Search
      │
      ▼
Relevant Document Chunks
      │
      ▼
Context + Question
      │
      ▼
       Gemini
      │
      ▼
Generated Answer
      │
      ▼
   Streamlit UI