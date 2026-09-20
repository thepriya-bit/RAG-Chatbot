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
    query_embedding = create_embeddings([question])[0]


    # -----------------------------------
    # 2. Retrieve relevant chunks
    # -----------------------------------
    results = search_documents(
        query_embedding,
        n_results=3
    )


    # -----------------------------------
    # 3. Combine retrieved chunks
    # -----------------------------------
    context = "\n\n".join(
        results["documents"][0]
    )


    # Debug: show retrieved context
    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("--- END CONTEXT ---\n")


    # -----------------------------------
    # 4. Create prompt for Gemini
    # -----------------------------------
    prompt = f"""
You are answering a question about a document.

The information you need is in the DOCUMENT CONTEXT below.
Answer using ONLY the context. Do not use outside knowledge.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Read the whole context carefully before answering.
- The answer may appear in a section with its own heading
  (for example 'INTERNSHIP DETAILS', 'PROJECTS' or 'EDUCATION'),
  and may not be written next to the person's name.
- Find the relevant text and answer directly, quoting or summarizing it.
- Only if no part of the context answers the question, say:
  "I could not find that information in the document."

ANSWER:
"""


    # -----------------------------------
    # 5. Send prompt to Gemini
    # -----------------------------------
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    # Debug: show Gemini response
    print("\n--- GEMINI RESPONSE ---")
    print(response.text)
    print("--- END RESPONSE ---\n")


    # -----------------------------------
    # 6. Return final answer
    # -----------------------------------
    return response.text