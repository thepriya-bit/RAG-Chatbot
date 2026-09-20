from embeddings import create_embeddings
from vector_store import search_documents


question = "What internships has Priya completed?"

query_embedding = create_embeddings([question])[0]

results = search_documents(query_embedding, n_results=1)

print("Retrieved document:")
print(results["documents"][0][0])