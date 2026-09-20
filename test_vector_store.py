from document_processor import load_pdf, chunk_text
from embeddings import create_embeddings
from vector_store import add_documents


pdf_path = "documents/Resume- PriyaSaikia.pdf"

text = load_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

add_documents(chunks, embeddings)

print("Documents successfully added to ChromaDB!")
print("Number of chunks:", len(chunks))