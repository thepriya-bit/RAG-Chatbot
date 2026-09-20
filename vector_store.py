import chromadb


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def clear_collection():
    global collection

    client.delete_collection("documents")

    collection = client.get_or_create_collection(
        name="documents"
    )


def add_documents(chunks, embeddings):
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )


def search_documents(query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count())
    )

    return results