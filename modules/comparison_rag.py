import re 

import streamlit as st 
from config import RETRIEVER_K

_CMP_KEYWORDS = [
    "so sánh", "khác nhau", "điểm khác", "điểm chung", "giống nhau",
    "tổng hợp", "liệt kê", "tương ứng", "thay đổi như thế nào",
    "biến đổi", r"tăng từ.{0,20}lên", r"giảm từ.{0,20}xuống",
    r"giai đoạn.{0,15}và.{0,15}giai đoạn",
    r"loại.{0,10}và.{0,10}loại",
    r"giữa.{0,30}và",
    "điểm nào", "khác biệt", "phân biệt", "đối chiếu",
    "compare", "difference", "similarity", "contrast", "versus", r"\bvs\b",
    r"how does.{0,40}differ", "what is the difference",
    "distinguish", "in common",
]

_CMP_PATTERN = re.compile('|'.join(_CMP_KEYWORDS), re.IGNORECASE)

_VI_CHARS = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"

def _dectect_language(text: str) -> str:
    return 'vi' if any(c in text.lower() for c in _VI_CHARS) else 'en'

def detect_comparison(question: str) -> bool:
    return bool(_CMP_PATTERN.search(question))

def decompose_query(question: str, llm) -> list[str]:
    lang = _dectect_language(question)

    if lang == 'vi':
        prompt = (
            "Bạn là hệ thống phân tách câu hỏi cho ứng dụng Q&A tài liệu.\n"
            "Câu hỏi dưới đây là câu so sánh hoặc tổng hợp nhiều đối tượng.\n"
            "Hãy tách thành 2-4 câu hỏi đơn giản, mỗi câu có thể tìm kiếm "
            "độc lập trong tài liệu.\n"
            "Chỉ xuất danh sách đánh số, mỗi câu một dòng, không giải thích thêm.\n\n"
            f"Câu hỏi gốc: {question}\n\n"
            "Các câu hỏi con:"
        )
    else:
        prompt = (
            "You are a query decomposer for a document Q&A system.\n"
            "The user asked a comparison/synthesis question. "
            "Break it into 2-4 simple, specific sub-questions that can each be "
            "answered by searching a document independently.\n"
            "Output ONLY a numbered list, one sub-question per line, no extra text.\n\n"
            f"Original question: {question}\n\n"
            "Sub-questions:"
        )

    try:
        raw = llm.invoke(prompt)
        text = raw.content.strip() if hasattr(raw, 'content') else str(raw).strip()
        lines = [
            re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip()
            for ln in text.splitlines()
            if ln.strip() and re.match(r"^\s*\d", ln)
        ]

        lines = [l for l in lines if len(l) > 8]
        return lines if lines else [question]
    except Exception:
        return [question]
    
def multi_retriever(sub_queries: list[str], retriever, k_per_query: int = RETRIEVER_K) -> list:
    seen = set()
    all_docs = []

    original_k = None
    search_kwargs = getattr(retriever, 'search_kwargs', None)
    if search_kwargs is not None:
        original_k = search_kwargs.get('k', RETRIEVER_K)
        search_kwargs['k'] = k_per_query

    for q in sub_queries:
        try:
            docs = retriever.invoke(q)
            for doc in docs:
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        except Exception:
            pass

    if search_kwargs is not None and original_k is not None:
        search_kwargs['k'] = original_k

    return all_docs

def _format_context(source_docs: list, max_chars: int = 7000) -> tuple[str, int]:
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(source_docs, 1):
        page = doc.metadata.get('page', '')
        source = doc.metadata.get('source', doc.metadata.get('source_-file', ''))
        labels = [f"Đoạn {i}"]
        if source:
            labels.append(f'File: {source}')
        if page != '':
            labels.append(f'Trang: {int(page) + 1}')
        chunk = f"[{' | '.join(labels)}]\n{doc.page_content}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total +=  len(chunk)
    return "\n\n---\n\n".join(parts), len(parts)


def _build_comparison_prompt(question: str, context: str) -> str:
    lang = _dectect_language(question)

    if lang == 'vi':
        return (
            "Bạn là chuyên gia phân tích tài liệu.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới. "
            "Mỗi đoạn được đánh nhãn [Đoạn X | Trang Y] — "
            "dùng đúng số liệu từ đúng đoạn, KHÔNG lẫn lộn giữa các đoạn.\n"
            "2. PHẢI trả lời HOÀN TOÀN bằng TIẾNG VIỆT. "
            "Tuyệt đối KHÔNG dùng tiếng Anh, tiếng Trung, tiếng Nga, "
            "tiếng Indonesia hay bất kỳ ngôn ngữ nào khác — kể cả một từ đơn lẻ.\n"
            "3. Nếu không tìm thấy thông tin trong ngữ cảnh, chỉ nói: "
            "'Tôi không tìm thấy thông tin này trong tài liệu.' "
            "KHÔNG bịa đặt hay suy diễn số liệu.\n"
            "4. Khi trích dẫn số liệu cụ thể, ghi rõ lấy từ Đoạn/Trang nào.\n\n"
            "ĐỊNH DẠNG CÂU TRẢ LỜI:\n"
            "- Mở đầu: 1 câu tóm tắt tổng quan về nội dung so sánh.\n"
            "- Thân bài: Ưu tiên dùng bảng Markdown (| Tiêu đề | ... |) "
            "khi so sánh 2+ đối tượng có cùng thuộc tính. "
            "Nếu không thể dùng bảng, dùng tiêu đề ## và gạch đầu dòng • nhất quán. "
            "KHÔNG trộn lẫn định dạng.\n"
            "- Kết luận: 1-2 câu tổng kết điểm chung / khác biệt quan trọng nhất.\n\n"
            f"NGỮ CẢNH TÀI LIỆU:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI (bằng tiếng Việt, đúng định dạng trên):"
        )
    else:
        return (
            "You are an expert document analyst.\n\n"
            "MANDATORY RULES:\n"
            "1. Use ONLY information from the CONTEXT section below. "
            "Each chunk is labeled [Chunk X | Page Y] — "
            "use figures from the correct chunk only, do NOT mix them up.\n"
            "2. Reply ENTIRELY in ENGLISH. "
            "Do NOT use Chinese, Russian, Vietnamese, Indonesian "
            "or any other language — not even a single word.\n"
            "3. If you cannot find the answer, say: "
            "'I cannot find this information in the document.' "
            "Do NOT fabricate or infer figures.\n"
            "4. When citing specific figures, mention the Chunk / Page.\n\n"
            "ANSWER FORMAT:\n"
            "- Opening: 1 sentence summarising what is being compared.\n"
            "- Body: Prefer a Markdown table (| Header | ... |) "
            "when comparing 2+ entities with shared attributes. "
            "Otherwise use ## headings and • bullets consistently. "
            "Do NOT mix formats.\n"
            "- Conclusion: 1-2 sentences on the most important "
            "similarities / differences.\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER (in English, following the format above):"
        )

def _md_table_to_html(md_table: str) -> str:
    rows = [r.strip() for r in md_table.strip().splitlines() if r.strip()]
    rows = [r for r in rows if not re.match(r"^\|[\s\-|:]+\|$", r)]
    if not rows:
        return md_table
    html_rows: list[str] = []
    for idx, row in enumerate(rows):
        cells = [c.strip() for c in row.strip('|').split('|')]
        if idx == 0:
            inner = ''.join(
                f'<th style="background:#0d2137;color:#7aaed0;padding:6px 10px;'
                f'border:1px solid #1e3a5f;text-align:left;">{c}</th>'
                for c in cells
            )
        else:
            bg = "#07111c" if idx % 2 == 0 else "#091520"
            inner = "".join(
                f'<td style="background:{bg};color:#c8e6c9;padding:5px 10px;'
                f'border:1px solid #1a2e45;">{c}</td>'
                for c in cells
            )
        html_rows.append(f"<tr>{inner}</tr>")
    return (
        '<div style="overflow-x:auto;margin:0.6rem 0;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.83rem;">'
        + "".join(html_rows)
        + "</table></div>"
    )

def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",       r"<code style='background:#0d2137;padding:1px 4px;"
                                    r"border-radius:3px;font-size:0.85em;'>\1</code>", text)
    return text

def format_answer_html(raw: str) -> str:
    lines = raw.strip().splitlines()
    html_parts: list[str] = []
    in_list = False
    in_table = False
    table_buf: list[str] = []

    def flush_list():
        nonlocal in_list
        if in_list:
            html_parts.append('</ul>')
            in_list = False

    def flush_table():
        nonlocal in_table, table_buf
        if in_table and table_buf:
            html_parts.append(_md_table_to_html('\n'.join(table_buf)))
            table_buf.clear()
            in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            flush_list()
            in_table = True
            table_buf.append(stripped)
            continue

        else:
            flush_table()

        if stripped.startswith('### '):
            flush_list()
            html_parts.append(
                f"<h5 style='margin:0.7rem 0 0.2rem;color:#7aaed0;"
                f"font-size:0.9rem;'>{_inline(stripped[4:])}</h5>"
            )
            continue

        if stripped.startswith('## '):
            flush_list()
            html_parts.append(
                f"<h4 style='margin:0.9rem 0 0.3rem;color:#0099ff;"
                f"font-size:1rem;border-bottom:1px solid #1e3a5f;"
                f"padding-bottom:0.2rem;'>{_inline(stripped[3:])}</h4>"
            )
            continue

        if stripped.startswith('# '):
            flush_list()
            html_parts.append(
                f"<h3 style='margin:1rem 0 0.35rem;color:#0099ff;"
                f"font-size:1.1rem;'>{_inline(stripped[2:])}</h3>"
            )
            continue

        bullet = re.match(r"^[\•\-\*]\s+(.+)$", stripped)

        if bullet:
            if not in_list:
                html_parts.append(
                    '<ul style="margin:0.25rem 0 0.25rem 1.1rem;'
                    'padding:0;line-height:1.75;color:#c8e6c9;">'
                )
                in_list = True
            html_parts.append(
                f"<li style='margin-bottom:0.15rem;'>{_inline(bullet.group(1))}</li>"
            )
            continue

        if not stripped:
            flush_list()
            continue

        flush_list()
        html_parts.append(
            f"<p style='margin:0.3rem 0;line-height:1.75;"
            f"color:#d0e8d0;'>{_inline(stripped)}</p>"
        )

    flush_list()
    flush_table()

    return '\n'.join(html_parts)

def comparison_rag_answer(question: str, retriever, llm) -> dict:
    lang = _dectect_language(question)
    sub_queries = decompose_query(question, llm)
    source_docs = multi_retriever(sub_queries, retriever, k_per_query=RETRIEVER_K)
    context, chunks_used = _format_context(source_docs, max_chars=7000)
    prompt = _build_comparison_prompt(question, context)
    raw_resp = llm.invoke(prompt)
    raw_text = raw_resp.content if hasattr(raw_resp, 'content') else str(raw_resp)

    return {
        'answer': raw_text,
        'answer_html': format_answer_html(raw_text), 
        'source_docs': source_docs[:10], 
        'sub_queries': sub_queries,
        'strategy': 'comparison_rag',
        'chunks_used': chunks_used,
        'lang': lang
    }

def render_comparison_tonggle():
    st.subheader('Comparison RAG')
    use_cmp = st.toggle(
        'Auto-detect comparison questions',
        value=st.session_state.get('use_comparison_auto', True),
        key='toggle_comparison_auto',
        help=(
            "ON -> Automatically compare recognition sentences, using multi-faceted retrieval"
            "OFF -> Always use path preparation"
        ),
    )
    st.session_state['use_comparison_auto'] = use_cmp
    if use_cmp:
        st.markdown(
            '<span style="font-size:0.75rem;color:#7aaed0;">'
            "Decompose → Multi-retrieve → Compare"
            "</span>",
            unsafe_allow_html=True,
        )

    return use_cmp

def render_comparison_metadata(result: dict):
    sub_qs = result.get('sub_queries', [])
    chunks = result.get('chunks_used', 0)
    lang = result.get('lang', 'vi')
    lbl_sq = 'Câu hỏi con' if lang == 'vi' else 'Sub-queries'
    lbl_ch = 'Chunks tổng hợp' if lang =='vi' else 'Chunks uesd'
    sub_html = ''.join(f'<li style="margin-bottom:0.2rem;color:#c8e6c9;">{q}</li>' for q in sub_qs)
    st.markdown(
        f"""
        <div style="
            background:#081510;
            border:1px solid #1a4a2a;
            border-left:3px solid #00cc66;
            border-radius:8px;
            padding:0.75rem 1rem;
            margin-top:0.6rem;
            font-size:0.8rem;
            line-height:1.6;
        ">
            <div style="
                color:#00cc66;
                font-family:'IBM Plex Mono',monospace;
                text-transform:uppercase;
                font-size:0.68rem;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;
            ">Comparison RAG</div>
 
            <div style="margin-bottom:0.4rem;">
                <span style="color:#7ad0a0;">{lbl_sq} ({len(sub_qs)}):</span>
                <ul style="margin:0.3rem 0 0 1rem;padding:0;">
                    {sub_html}
                </ul>
            </div>
 
            <div>
                <span style="color:#7ad0a0;">{lbl_ch}:</span>
                <span style="color:#00cc66;font-weight:700;
                             margin-left:0.4rem;">{chunks}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
