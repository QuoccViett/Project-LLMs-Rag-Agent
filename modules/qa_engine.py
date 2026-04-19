def _detect_language(text: str) -> str:
    vi_chart = "àáảãạăắặằẳẵâấậầẩẫđèéẻẽẹêếệềểễìíỉĩịòóỏõọôốộồổỗơớợờởỡùúủũụưứựừửữỳýỷỹỵ"
    return 'vi' if any(c in text.lower() for c in vi_chart) else 'en'

def build_prompt(context: str, question: str) -> str:
    lang = _detect_language(question)

    if lang == 'vi':
        return (
            "Bạn là một chuyên gia phân tích tài liệu (ADI System). "
            "Hãy sử dụng thông tin từ các tài liệu được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác nhất.\n\n"
            "YÊU CẦU QUAN TRỌNG:\n"
            "1. CHỈ sử dụng thông tin trong Context được cung cấp. Không tự suy diễn.\n"
            "2. Trả lời chi tiết, có cấu trúc (sử dụng gạch đầu dòng nếu cần).\n"
            "3. TRẢ LỜI HOÀN TOÀN BẰNG TIẾNG VIỆT. Tuyệt đối không sử dụng tiếng Trung hay tiếng Anh.\n"
            "4. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung, chữ Hán hoặc bất kỳ ngôn ngữ nào khác ngoài tiếng Việt.\n"
            "5. Nếu thông tin không có trong context, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
            f"--- CONTEXT ---\n{context}\n\n"
            f"--- CÂU HỎI ---\n{question}\n\n"
            "--- TRẢ LỜI ---"
        )
    return (
        "You are an Advanced Document Intelligence (ADI) expert. "
        "Use the following context to provide a professional and detailed answer.\n\n"
        "RULES:\n"
        "1. Answer based ONLY on the provided context.\n"
        "2. If the answer is missing, say 'Information not found in documents.'\n"
        "3. Reply in a structured and clear manner.\n\n"
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