from rag_pipeline import answer_question


question = "What is Priya's date of birth?"

answer = answer_question(question)

print("Question:", question)
print("\nAnswer:", answer)