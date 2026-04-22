import streamlit as st
import re 
from config import CONV_MEMORY_K, RETRIEVER_K
from langchain_core.messages import HumanMessage, AIMessage
from modules.qa_engine import _format_chunks_with_source, _needs_calculation, _mentions_multiple_pages

def _is_followup(question: str, memory: list) -> bool:
    if not memory:
        return False
    followup_starters = (
        'it', 'its', 'they', 'their', 'them',
        'that', 'this', 'those', 'these',
        'he', 'she', 'what about', 'and', 'also',
        'why', 'how about', 'what else',
        'nó', 'họ', 'còn', 'vậy', 'tại sao', 'thế',
        'còn về', 'vậy thì', 'thế còn', 'cái đó', 'điều đó',
    )
    q_lower = question.strip().lower()
    return any(q_lower.startswith(w) for w in followup_starters)

def _format_memory(memory: list) -> str:
    lines = []
    for turn in memory:
        role = 'Người dùng' if turn.type == 'human' else 'Trợ lý'
        content = turn.content[:300]
        if len(turn.content) > 300:
            content += '...'
        lines.append(f'{role}: {content}')
    return '\n'.join(lines)

def _detect_language(text: str) -> str:
    vi_chart = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    return 'vi' if any(c in text.lower() for c in vi_chart) else 'en'


def _condense_question(question: str, memory: list, llm) -> str:
    if not _is_followup(question, memory):
        return question
    history_text = _format_memory(memory)
    lang = _detect_language(question)
    if lang == 'vi':
        condensation_prompt = (
            "Bạn là trợ lý ngôn ngữ. Nhiệm vụ: đọc lịch sử trò chuyện và câu hỏi mới, "
            "sau đó viết lại thành một câu hỏi ĐỘC LẬP, đầy đủ ý nghĩa bằng TIẾNG VIỆT.\n"
            "Câu hỏi viết lại phải bao gồm đủ danh từ riêng, chủ thể, trang/mục cụ thể "
            "để có thể tìm kiếm trong tài liệu mà không cần xem lại lịch sử.\n"
            "Yêu cầu: CHỈ trả về câu hỏi đã viết lại. KHÔNG giải thích. "
            "KHÔNG dùng tiếng Trung, tiếng Nga hay tiếng Anh.\n\n"
            f"Lịch sử trò chuyện:\n{history_text}\n\n"
            f"Câu hỏi mới: {question}\n"
            "Câu hỏi độc lập (tiếng Việt):"
        )
    else:
        condensation_prompt = (
            "You are a linguistic assistant. Task: read the chat history and the new question, "
            "then rewrite it into a STANDALONE question in ENGLISH that is fully self-contained.\n"
            "Include all necessary proper nouns, page references, and context so it can be "
            "searched in a document without referring back to the history.\n"
            "Requirements: return ONLY the rewritten question. NO explanation. "
            "NO Chinese, Russian or Vietnamese.\n\n"
            f"Chat history:\n{history_text}\n\n"
            f"New question: {question}\n"
            "Standalone question (English):"
        )
    
    try:
        response = llm.invoke(condensation_prompt)
        condensed = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if not condensed or len(condensed) < 2:
            return question
        return condensed
    except Exception:
        return question

def _build_conv_prompt(context: str, question: str, memory: list, source_docs: list) -> str:
    history_section = ''
    lang = _detect_language(question)
    needs_calc = _needs_calculation(question)
    context = _format_chunks_with_source(source_docs)
    if needs_calc:
        calc_instruction_vi = (
            "\n5. Câu hỏi này yêu cầu tính toán, so sánh hoặc đánh giá ngưỡng:\n"
            "   - Bước 1: Liệt kê từng số liệu liên quan kèm nguồn (Đoạn mấy, Trang mấy).\n"
            "   - Bước 2: Thực hiện phép tính hoặc so sánh từng bước.\n"
            "   - Bước 3: Đưa ra kết luận rõ ràng (ví dụ: 'Vượt ngưỡng', 'Cần thực hiện ngay', v.v.)."
        )
        calc_instruction_en = (
            "\n5. This question requires calculation, comparison, or threshold evaluation:\n"
            "   - Step 1: List each relevant figure with its source (Chunk number, Page number).\n"
            "   - Step 2: Perform the calculation or comparison step by step.\n"
            "   - Step 3: State a clear conclusion (e.g. 'Exceeds threshold', 'Action required', etc.)."
        )
    else:
        calc_instruction_vi = ""
        calc_instruction_en = ""
    if lang == 'vi':
        history_section = (
            "Lịch sử hội thoại gần nhất:\n"
                + _format_memory(memory[-(CONV_MEMORY_K * 2):])
                + '\n\n'
        ) if memory else ''
        
        return (
            "Bạn là trợ lý AI thông minh. Trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "Quy tắc bắt buộc:\n"
            "1. Chỉ dùng thông tin trong phần Ngữ cảnh tài liệu. "
            "Mỗi đoạn được đánh nhãn [Đoạn X | Trang Y] — "
            "hãy dùng đúng số liệu từ đúng trang, "
            "KHÔNG được lẫn lộn số liệu giữa các trang khác nhau.\n"
            "2. NGÔN NGỮ — QUY TẮC TUYỆT ĐỐI:\n"
            "   PHẢI viết HOÀN TOÀN bằng TIẾNG VIỆT.\n"
            "   CẤM TUYỆT ĐỐI dùng: tiếng Anh, tiếng Trung (中文/汉字), tiếng Nga,\n"
            "   tiếng Indonesia, tiếng Nhật hay bất kỳ ngôn ngữ nào khác.\n"
            "   Ví dụ SAI: '根据文件' hoặc 'According to' hoặc 'Berdasarkan'\n"
            "   Ví dụ ĐÚNG: 'Theo tài liệu' hoặc 'Dựa trên tài liệu'\n"
            "3. Nếu không tìm thấy thông tin, nói: "
            "'Tôi không tìm thấy thông tin này trong tài liệu.' "
            "KHÔNG được tự bịa đặt hay suy diễn thêm số liệu.\n"
            "4. Khi trích dẫn số liệu cụ thể, hãy ghi rõ lấy từ Trang nào."
            f"{calc_instruction_vi}\n\n"
            f"Ngữ cảnh tài liệu:\n{context}\n\n"
            f"{history_section}"
            f"Người dùng: {question}\n\n"
            "Trợ lý (viết hoàn toàn bằng tiếng Việt, không dùng bất kỳ ngôn ngữ nào khác):"
        )
    else:
        history_section = (
            "Recent chat history:\n"
            + _format_memory(memory[-(CONV_MEMORY_K * 2):])
            + '\n\n'
        ) if memory else ''

        return (
            "You are a helpful AI assistant. Answer questions based on the document context.\n"
            "Rules:\n"
            "1. Use ONLY information from the Document context section. "
            "Each chunk is labeled [Chunk X | Page Y] — "
            "use figures from the correct page only, "
            "do NOT mix up figures between different pages.\n"
            "2. LANGUAGE — ABSOLUTE RULE:\n"
            "   Reply ENTIRELY in ENGLISH.\n"
            "   FORBIDDEN: Chinese (中文), Russian, Vietnamese, Indonesian, Japanese\n"
            "   or any other language — not even a single word.\n"
            "   WRONG example: '根据文件' or 'Theo tài liệu'\n"
            "   RIGHT example: 'According to the document'\n"
            "3. If you cannot find the answer, say: "
            "'I cannot find this information in the document.' "
            "Do NOT fabricate or infer figures.\n"
            "4. When citing specific figures, mention which Page they come from."
            f"{calc_instruction_en}\n\n"
            f"Document context:\n{context}\n\n"
            f"{history_section}"
            f"User: {question}\n\n"
            "Assistant (English only, absolutely no other language):"
        )

def get_answer_with_memory(question: str, retriever, llm) -> tuple[str, list]:
    if 'conv_memory' not in st.session_state:
        st.session_state.conv_memory =[]

    memory = st.session_state.conv_memory

    standalone = _condense_question(question, memory, llm)
    source_docs = retriever.invoke(standalone)

    if _mentions_multiple_pages(standalone):
        page_mentions = re.findall(r'trang\s*\d+|page\s*\d+', standalone.lower())
        needed_k = max(len(page_mentions) * 2, RETRIEVER_K)
        search_kwargs = getattr(retriever, 'search_kwargs', None)
        if search_kwargs is not None:
            original_k = retriever.search_kwargs.get('k', RETRIEVER_K)
            search_kwargs['k'] = needed_k
            source_docs = retriever.invoke(standalone)
            search_kwargs['k'] = original_k
        else:
            source_docs = retriever.invoke(standalone)
    else:
        source_docs = retriever.invoke(standalone)

    # context = '\n\n'.join(doc.page_content for doc in source_docs)
    prompt = _build_conv_prompt('', question, memory, source_docs)
    answer = llm.invoke(prompt)
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
    memory.append(HumanMessage(content=question))
    memory.append(AIMessage(content=answer_text))
    st.session_state.conv_memory = memory[-(CONV_MEMORY_K * 2):]

    return answer, source_docs

def render_memory_badge():
    if 'conv_memory' not in st.session_state:
        return 
    n = len(st.session_state.conv_memory) // 2
    if n:
        st.markdown(
            f'<span style="font-size:0.75rem;color:#7aaed0;">'
            f'Memory: <b>{n}</b> turn{"s" if n > 1 else ""} retained</span>',
            unsafe_allow_html=True,
        )
