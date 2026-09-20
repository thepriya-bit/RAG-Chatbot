import os

from dotenv import load_dotenv
from google import genai

from embeddings import create_embeddings
from vector_store import search_documents


# Load environment variables
load_dotenv()


# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def answer_question(question, chat_history=None):

    # -----------------------------------
    # 1. Convert question into embedding
    # -----------------------------------

    query_embedding = create_embeddings(
        [question]
    )[0]


    # -----------------------------------
    # 2. Retrieve relevant chunks
    # -----------------------------------

    results = search_documents(
        query_embedding,
        n_results=3
    )


    # -----------------------------------
    # 3. Get documents and metadata
    # -----------------------------------

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]


    # -----------------------------------
    # 4. Combine chunks with their sources
    # -----------------------------------

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        filename = metadata["filename"]

        context_parts.append(
            f"""
SOURCE: {filename}

CONTENT:
{document}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # -----------------------------------
    # Debug: show retrieved context
    # -----------------------------------

    print("\n--- RETRIEVED CONTEXT WITH SOURCES ---")
    print(context)
    print("--- END CONTEXT ---\n")


    # -----------------------------------
    # 5. Create prompt for Gemini
    # -----------------------------------

    prompt = f"""
You are answering a question about uploaded documents.

Use ONLY the information provided in the DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer the question using the provided document context.
- Read the relevant content carefully before answering.
- The answer may appear under a section heading and may not
  be directly next to the person's name.
- Do not use outside knowledge.
- Do not invent information.
- If the answer is present in the context, answer it clearly.
- If the answer is not present in the context, say:
  "I could not find that information in the uploaded documents."

ANSWER:
"""


    # -----------------------------------
    # 6. Send prompt to Gemini
    # -----------------------------------

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    # -----------------------------------
    # Debug: show Gemini response
    # -----------------------------------

    print("\n--- GEMINI RESPONSE ---")
    print(response.text)
    print("--- END RESPONSE ---\n")


    # -----------------------------------
    # 7. Return answer
    # -----------------------------------

    return response.text