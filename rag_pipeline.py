import os

from dotenv import load_dotenv
from google import genai

from embeddings import create_embeddings
from vector_store import search_documents


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def answer_question(question):
    query_embedding = create_embeddings([question])[0]

    results = search_documents(
        query_embedding,
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using only the information provided in the context.

If the answer cannot be found in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text