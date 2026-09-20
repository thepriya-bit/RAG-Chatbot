from rag_pipeline import answer_question


question = "What internships has Priya completed?"

answer = answer_question(question)

print("Question:", question)
print("\nAnswer:", answer)