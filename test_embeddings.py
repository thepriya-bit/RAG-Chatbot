from embeddings import create_embeddings


texts = [
    "Priya completed an internship in artificial intelligence and machine learning.",
    "Priya worked on an NLP and digital preservation project."
]

embeddings = create_embeddings(texts)

print("Number of embeddings:", len(embeddings))
print("Dimensions of each embedding:", len(embeddings[0]))