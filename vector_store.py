import chromadb


client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


def clear_collection():
    global collection

    client.delete_collection("documents")

    collection = client.get_or_create_collection(
        name="documents"
    )


def add_documents(chunks, embeddings, filename, document_id):

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": filename,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def search_documents(query_embedding, n_results=3):

    document_count = collection.count()

    if document_count == 0:
        return {
            "documents": [[]],
            "metadatas": [[]]
        }

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, document_count)
    )

    return results