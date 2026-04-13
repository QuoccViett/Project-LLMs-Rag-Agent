def _detect_language(text: str) -> str:
    vi_chart = "àáảãạăắặằẳẵâấậầẩẫđèéẻẽẹêếệềểễìíỉĩịòóỏõọôốộồổỗơớợờởỡùúủũụưứựừửữỳýỷỹỵ"
    return 'vi' if any(c in text.lower() for c in vi_chart) else 'en'

def build_prompt(context: str, question: str) -> str:
    lang = _detect_language(question)

    if lang == 'vi':
        return (
            "Use the context below to answer the question. "
            "If you don't know, say no. Reply concisely (3-4 sentences) in Vietnamese.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    return (
        "Use the context below to answer the question. "
            "If you don't know, say no. Keep the answer concise (3-4 sentences).\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
    )

def get_answer(question: str, retriever, llm) -> tuple[str, list]:
    source_docs = retriever.invoke(question)
    context = '\n\n'.join(doc.page_content for doc in source_docs)
    prompt = build_prompt(context, question)
    answer = llm.invoke(prompt)
    return answer, source_docs