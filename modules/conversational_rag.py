import streamlit as st 
from config import CONV_MEMORY_K
from langchain_core.messages import HumanMessage, AIMessage

def _is_followup(question: str, memory: list) -> bool:
    if not memory:
        return False
    followup_starters = (
        'it', 'its', 'they', 'their', 'them',
        'that', 'this', ' those', 'these',
        'he', 'she', 'what about', 'and', 'also',
        'why', 'how about', 'what else',
        'nó', 'họ', 'còn', 'vậy', 'tại sao', 'thế'
    )
    q_lower = question.strip().lower()
    return any(q_lower.startswith(w) for w in followup_starters)

def _format_memory(memory: list) -> str:
    lines = []
    for turn in memory:
        role = 'User' if turn.type == 'human' else 'Assistant'
        content = turn.content[:300]
        if len(turn.content) > 300:
            content += '...'
        lines.append(f'{role}: {content}')
    return '\n'.join(lines)

def _detect_language(text: str) -> str:
    vi_chart = "àáảãạăắặằẳẵâấậầẩẫđèéẻẽẹêếệềểễìíỉĩịòóỏõọôốộồổỗơớợờởỡùúủũụưứựừửữỳýỷỹỵ"
    return 'vi' if any(c in text.lower() for c in vi_chart) else 'en'


def _condense_question(question: str, memory: list, llm) -> str:
    if not _is_followup(question, memory):
        return question
    history_text = _format_memory(memory)
    lang = _detect_language(question)
    if lang == 'vi':
        condensation_prompt = (
            "Bạn là một trợ lý ngôn ngữ chuyên nghiệp. Nhiệm vụ của bạn là đọc lịch sử trò chuyện "
            "và câu hỏi mới, sau đó viết lại thành một câu hỏi ĐỘC LẬP bằng TIẾNG VIỆT.\n"
            "Câu hỏi mới này phải đầy đủ ý nghĩa, bao gồm các danh từ riêng (tên người, tổ chức, mốc thời gian) "
            "để có thể tìm kiếm trong tài liệu mà không cần xem lại lịch sử.\n"
            "Yêu cầu: KHÔNG giải thích, KHÔNG dùng tiếng Trung/Anh, chỉ trả về câu hỏi đã viết lại.\n\n"
            f"Lịch sử:\n{history_text}\n"
            f"Câu hỏi mới: {question}\n"
            "Câu hỏi độc lập:"
        )
    else:
        condensation_prompt = (
            "You are a professional linguistic assistant. Your task is to read the chat history "
            "and the new question, then rewrite it into a STANDALONE question in VIETNAMESE.\n"
            "This new question must be fully meaningful, including proper nouns (names of people, organizations, timestamps) "
            "so that it can be searched in documents without needing to refer back to the history.\n"
            "Requirements: NO explanation, NO Chinese/English usage, return only the rewritten question.\n\n"
            f"History:\n{history_text}\n"
            f"New question: {question}\n"
            "Standalone question:"
        )
    
    try:
        response = llm.invoke(condensation_prompt)
        condensed = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if not condensed or len(condensed) < 2:
            return question
        return condensed
    except Exception:
        return question

def _build_conv_prompt(context: str, question: str, memory: list) -> str:
    history_section = ''
    lang = _detect_language(question)
    if memory:
        history_section = (
            'Chat history (most recent turns):\n'
            + _format_memory(memory[-(CONV_MEMORY_K * 2):])
            + '\n\n'
        )

    if lang == 'vi':
        return (
            "Bạn là một trợ lý ảo thông minh. Hãy sử dụng ngữ cảnh từ tài liệu dưới đây để trả lời câu hỏi. "
            "Nếu thông tin không có trong tài liệu, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.\n\n"
            f"Ngữ cảnh tài liệu:\n{context}\n\n"
            f"{history_section}"
            f"Người dùng: {question}\n\n"
            "Trợ lý (Trả lời bằng tiếng Việt):"
        )
    else:
        return (
            "You are a helpful assistant. Use the document context below to answer "
            "the user's question. If you cannot find the answer in the context, say so.\n\n"
            f"Document context:\n{context}\n\n"
            f"{history_section}"
            f"User: {question}\n\n"
            "Assistant:"
        )

def get_answer_with_memory(question: str, retriever, llm) -> tuple[str, list]:
    if 'conv_memory' not in st.session_state:
        st.session_state.conv_memory =[]

    memory = st.session_state.conv_memory

    standalone = _condense_question(question, memory, llm)
    source_docs = retriever.invoke(standalone)
    context = '\n\n'.join(doc.page_content for doc in source_docs)
    prompt = _build_conv_prompt(context, question, memory)
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
