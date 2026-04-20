import re
from config import RETRIEVER_K

def _detect_language(text: str) -> str:
    vi_chart = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    return 'vi' if any(c in text.lower() for c in vi_chart) else 'en'

def _needs_calculation(question: str) -> bool:
    calc_keywords = [
        'tổng', 'cộng', 'trừ', 'tính', 'bằng bao nhiêu', 'bằng nhau', 'so sánh',
        'tăng', 'giảm', 'chênh lệch', 'nhiều hơn', 'ít hơn', 'lớn hơn', 'nhỏ hơn',
        'vượt', 'vượt quá', 'dưới', 'trên', 'ngưỡng', 'cảnh báo', 'có cần', 'cần không',
        'có đạt', 'có bằng', 'có phải',
        'sum', 'total', 'add', 'subtract', 'compare', 'difference', 'calculate',
        'more than', 'less than', 'equal', 'exceed', 'threshold', 'warning',
        'should', 'need to', 'required',
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in calc_keywords)

def _mentions_multiple_pages(question: str) -> bool:
    vi_pages = re.findall(r'trang\s*\d+', question.lower())
    en_pages = re.findall(r'page\s*\d+', question.lower())
    return len(vi_pages) >= 2 or len(en_pages) >= 2

def _format_chunks_with_source(source_docs: list) -> str:
    parts = []
    for i, doc in enumerate(source_docs, 1):
        page  = doc.metadata.get('page', '')
        source = doc.metadata.get('source', doc.metadata.get('source_file', ''))
        label_parts = [f'[Đoạn {i}]']
        if source:
            label_parts.append(f'File: {source}')
        if page != '':
            label_parts.append(f'Trang: {int(page) + 1}')
        label = ' | '.join(label_parts)
        parts.append(f'{label}\n{doc.page_content}')
    return '\n\n---\n\n'.join(parts)

def build_prompt(context: str, question: str, source_docs: list = None) -> str:
    lang = _detect_language(question)
    needs_calc = _needs_calculation(question)

    if source_docs:
        context = _format_chunks_with_source(source_docs)

    if lang == 'vi':
        calc_instruction = (
            "\n5. Câu hỏi này yêu cầu tính toán, so sánh hoặc đánh giá ngưỡng:\n"
            "   - Bước 1: Liệt kê từng số liệu liên quan kèm nguồn (Đoạn mấy, Trang mấy).\n"
            "   - Bước 2: Thực hiện phép tính hoặc so sánh từng bước.\n"
            "   - Bước 3: Đưa ra kết luận rõ ràng (ví dụ: 'Vượt ngưỡng', 'Cần thực hiện ngay', v.v.)."
            if needs_calc else ""
        )
        return (
            "Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu.\n"
            "Quy tắc bắt buộc:\n"
            "1. Chỉ sử dụng thông tin có trong phần Ngữ cảnh bên dưới. "
            "Mỗi đoạn được đánh nhãn [Đoạn X | Trang Y] — "
            "hãy dùng đúng số liệu từ đúng trang, "
            "KHÔNG được lẫn lộn số liệu giữa các trang khác nhau.\n"
            "2. PHẢI trả lời hoàn toàn bằng TIẾNG VIỆT. "
            "Tuyệt đối KHÔNG dùng tiếng Anh, tiếng Trung, tiếng Nga, tiếng Indonesia "
            "hay bất kỳ ngôn ngữ nào khác dù chỉ một từ.\n"
            "3. Nếu không tìm thấy thông tin trong ngữ cảnh, chỉ nói: "
            "'Tôi không tìm thấy thông tin này trong tài liệu.' "
            "KHÔNG được tự bịa đặt hay suy diễn thêm số liệu.\n"
            "4. Khi trích dẫn số liệu cụ thể, hãy ghi rõ lấy từ Trang nào."
            f"{calc_instruction}\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời (bằng tiếng Việt):"
        )
    else:
        calc_instruction = (
            "\n5. This question requires calculation, comparison, or threshold evaluation:\n"
            "   - Step 1: List each relevant figure with its source (Chunk number, Page number).\n"
            "   - Step 2: Perform the calculation or comparison step by step.\n"
            "   - Step 3: State a clear conclusion (e.g. 'Exceeds threshold', 'Action required', etc.)."
            if needs_calc else ""
        )
        return (
            "You are an AI assistant that answers questions based strictly on the provided document.\n"
            "Rules:\n"
            "1. Use ONLY the information in the Context section below. "
            "Each chunk is labeled [Chunk X | Page Y] — "
            "use figures from the correct page only, "
            "do NOT mix up figures between different pages.\n"
            "2. Reply in ENGLISH only. "
            "Do NOT use Chinese, Russian, Vietnamese, Indonesian "
            "or any other language — not even a single word.\n"
            "3. If you cannot find the answer, say: "
            "'I cannot find this information in the document.' "
            "Do NOT fabricate or infer figures.\n"
            "4. When citing specific figures, mention which Page they come from."
            f"{calc_instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

def get_answer(question: str, retriever, llm) -> tuple[str, list]:
    if _mentions_multiple_pages(question):
        page_mentions = re.findall(r'trang\s*\d+|page\s*\d+', question.lower())
        needed_k = max(len(page_mentions) * 2, RETRIEVER_K)

        try:
            original_k = retriever.search_kwargs.get('k', RETRIEVER_K)
            retriever.search_kwrags['k'] = needed_k
            source_docs = retriever.invoke(question)
            retriever.search_kwargs['k'] = original_k
        except AttributeError:
            source_docs = retriever.invoke(question)
    else:
        source_docs = retriever.invoke(question)

    # context_text  = '\n\n'.join(doc.page_content for doc in source_docs)
    prompt = build_prompt('', question, source_docs=source_docs)
    answer = llm.invoke(prompt)
    return answer, source_docs