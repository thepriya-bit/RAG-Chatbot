from embeddings import create_embeddings
from vector_store import search_documents


question = "What technologies were used to build JolBondhu?"

query_embedding = create_embeddings(
    [question]
)[0]

results = search_documents(
    query_embedding,
    n_results=2
)


print("\n--- RETRIEVED RESULTS ---")

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):

    print("\nSource:", metadata["filename"])
    print("Content:", document)