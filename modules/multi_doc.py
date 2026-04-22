import os 
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
from langchain_core.messages import HumanMessage, AIMessage

MULTI_DOC_RETRIEVER_K = max(RETRIEVER_K * 2, 20)

def _detect_language(text: str) -> bool:
    vi_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    return 'vi' if any(c in text.lower() for c in vi_chars) else 'en'


def _need_calculations(question: str) -> bool:
    calc_keywords = [
        'tổng', 'cộng', 'trừ', 'tính', 'bằng bao nhiêu', 'so sánh',
        'tăng', 'giảm', 'chênh lệch', 'nhiều hơn', 'ít hơn', 'lớn hơn', 'nhỏ hơn',
        'vượt', 'vượt quá', 'dưới', 'trên', 'ngưỡng', 'lệch', 'có đúng', 'có phải',
        'sum', 'total', 'compare', 'difference', 'calculate', 'more than', 'less than',
        'exceed', 'threshold', 'correct', 'verify',
    ]
    return any(kw in question.lower() for kw in calc_keywords)

def _format_multidoc_chunks(source_docs: list) -> str:
    parts = []
    for i, doc in enumerate(source_docs, 1):
        page = doc.metadata.get('page', '')
        source_file = doc.metadata.get('source_file', doc.metadata.get('source', ''))
        label_parts = [f'[Đoạn {i}]']
        if source_file:
            label_parts.append(f'File: {source_file}')
        if page != '':
            label_parts.append(f'Trang: {int(page) + 1}')
        label = ' | '.join(label_parts)
        parts.append(f'{label}\n{doc.page_content}')
    return '\n\n---\n\n'.join(parts)


def _list_loaded_files() -> str:
    registry = st.session_state.get('doc_registry', {})
    if not registry:
        return ''
    names = list(registry.keys())
    return ', '.join(f'"{n}"' for n in names)

def build_multidoc_prompt(question: str, source_docs: list) -> str:
    lang = _detect_language(question)
    context = _format_multidoc_chunks(source_docs)
    needs_calc = _need_calculations(question)
    loaded_files = _list_loaded_files()

    cross_kw_vi = [
        'hệ thống chính', 'hệ thống dự phòng', 'so sánh', 'đối chiếu',
        'cả hai', 'cả ba', 'file nào', 'tài liệu nào', 'hai tài liệu',
        'và hệ thống', 'còn hệ thống', 'quy trình khẩn cấp', 'bảo trì',
        'nhật ký', 'có đúng', 'có phải', 'đúng quy định', 'giữa các', 'khác nhau',
        'điểm chung', 'điểm khác', 'liệt kê', 'tổng hợp',
    ]

    cross_kw_en = [
        'main system', 'backup system', 'compare', 'both', 'which file',
        'cross-reference', 'maintenance', 'log', 'correct', 'verify', 'compliant',
        'difference', 'similarity', 'across', 'all documents', 'each file',
    ]

    q_lower = question.lower()
    is_cross = any(kw in q_lower for kw in (cross_kw_vi + cross_kw_en))

    cross_note_vi = (
        "\n5. Câu hỏi liên quan đến NHIỀU tài liệu:\n"
        f"   - Các file đã tải: {loaded_files}\n"
        "   - Phân biệt rõ thông tin từ file nào (nhãn 'File:' ở mỗi đoạn).\n"
        "   - Trình bày riêng từng file khi so sánh hoặc đối chiếu.\n"
        "   - KHÔNG nhầm lẫn dữ liệu giữa các file.\n"
        if is_cross else (
            f"   [Tài liệu đang có: {loaded_files}]\n" if loaded_files else ""
        )
    )

    cross_note_en = (
        "\n5. This question involves MULTIPLE documents:\n"
        f"   - Loaded files: {loaded_files}\n"
        "   - Distinguish which file each piece of information comes from (see 'File:' label).\n"
        "   - Present each file separately when comparing or cross-referencing.\n"
        "   - Do NOT mix up data between files.\n"
        if is_cross else (
            f"   [Available documents: {loaded_files}]\n" if loaded_files else ""
        )
    )

    calc_vi = (
        "\n6. Câu hỏi yêu cầu tính toán hoặc xác minh:\n"
        "   - Bước 1: Liệt kê từng số liệu liên quan kèm nguồn (File, Trang).\n"
        "   - Bước 2: Thực hiện phép tính hoặc so sánh từng bước.\n"
        "   - Bước 3: Đưa ra kết luận rõ ràng.\n"
        if needs_calc else ""
    )

    calc_en = (
        "\n6. This question requires calculation or verification:\n"
        "   - Step 1: List each relevant figure with its source (File, Page).\n"
        "   - Step 2: Perform the calculation or comparison step by step.\n"
        "   - Step 3: State a clear conclusion.\n"
        if needs_calc else ""
    )

    if lang == 'vi':
        return (
            "Bạn là trợ lý AI chuyên phân tích đa tài liệu.\n\n"
            "═══ QUY TẮC BẮT BUỘC (VI PHẠM = SAI) ═══\n"
            "1. CHỈ dùng thông tin trong phần [NGỮ CẢNH] bên dưới. "
            "Mỗi đoạn được đánh nhãn rõ ràng:\n"
            "   [Đoạn X | File: tên_file | Trang: Y]\n"
            "   KHÔNG lẫn lộn số liệu giữa các trang hoặc giữa các file khác nhau.\n"
            "   KHÔNG dùng kiến thức bên ngoài tài liệu.\n\n"
            "2. TRÍCH DẪN NGUỒN TRONG TỪNG CÂU — BẮT BUỘC:\n"
            "   Mỗi khi nêu thông tin cụ thể, phải ghi ngay sau đó:\n"
            "   (Nguồn: File [tên file], Trang [số trang])\n"
            "   Ví dụ: 'Tổng doanh thu là 500 triệu đồng (Nguồn: File bao_cao.pdf, Trang 3).'\n"
            "   KHÔNG được nêu thông tin mà không có trích dẫn nguồn.\n\n"
            "3. NGÔN NGỮ — QUY TẮC TUYỆT ĐỐI:\n"
            "   PHẢI viết HOÀN TOÀN bằng TIẾNG VIỆT.\n"
            "   CẤM TUYỆT ĐỐI: tiếng Trung (中文/汉字), tiếng Anh, tiếng Nga,\n"
            "   tiếng Indonesia, tiếng Nhật hay bất kỳ ngôn ngữ nào khác.\n"
            "   Ví dụ SAI: '根据文件' hoặc 'According to'\n"
            "   Ví dụ ĐÚNG: 'Theo tài liệu' hoặc 'Dựa trên tài liệu'\n\n"
            "4. Nếu ngữ cảnh KHÔNG có thông tin cho câu hỏi, chỉ nói:\n"
            "   'Tôi không tìm thấy thông tin này trong các tài liệu được cung cấp.'\n"
            "   KHÔNG bịa đặt, suy diễn hay thêm thông tin ngoài ngữ cảnh.\n"
            f"{cross_note_vi}"
            f"{calc_vi}\n"
            "═══════════════════════════════════════════\n\n"
            f"[NGỮ CẢNH từ các tài liệu]\n{context}\n\n"
            "═══════════════════════════════════════════\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời (bắt buộc tiếng Việt, mỗi thông tin phải có trích dẫn nguồn):"
        )
    else:
        return (
            "You are an AI assistant specializing in multi-document analysis.\n\n"
            "═══ MANDATORY RULES (VIOLATION = INCORRECT) ═══\n"
            "1. Use ONLY information from the [CONTEXT] section below. "
            "Each chunk is clearly labeled:\n"
            "   [Chunk X | File: filename | Page: Y]\n"
            "   Do NOT mix figures between different pages or different files.\n"
            "   Do NOT use any knowledge outside the provided documents.\n\n"
            "2. INLINE CITATION IN EVERY SENTENCE — MANDATORY:\n"
            "   After each specific piece of information, immediately add:\n"
            "   (Source: File [filename], Page [number])\n"
            "   Example: 'Total revenue is $500,000 (Source: File report.pdf, Page 3).'\n"
            "   NEVER state information without citing the source.\n\n"
            "3. LANGUAGE — ABSOLUTE RULE:\n"
            "   Reply ENTIRELY in ENGLISH.\n"
            "   FORBIDDEN: Chinese (中文/汉字), Vietnamese, Russian, Indonesian, Japanese\n"
            "   or any other language — not even a single character.\n"
            "   WRONG: '根据文件' or 'Theo tài liệu'\n"
            "   RIGHT: 'According to the document'\n\n"
            "4. If the context does NOT contain information for the question, say:\n"
            "   'I cannot find this information in the provided documents.'\n"
            "   Do NOT fabricate, infer, or add information beyond the context.\n"
            f"{cross_note_en}"
            f"{calc_en}\n"
            "═══════════════════════════════════════════\n\n"
            f"[CONTEXT from documents]\n{context}\n\n"
            "═══════════════════════════════════════════\n\n"
            f"Question: {question}\n\n"
            "Answer (English only, every fact must have inline source citation):"
        )
    
def get_multidoc_answer(question: str, retriever, llm) -> tuple:
    search_kwargs = getattr(retriever, 'search_kwargs', None)
    original_k = None
    if search_kwargs is not None:
        original_k = search_kwargs.get('k', RETRIEVER_K)
        search_kwargs['k'] = MULTI_DOC_RETRIEVER_K

    source_docs = retriever.invoke(question)

    if search_kwargs is not None and original_k is not None:
        search_kwargs['k'] = original_k

    prompt = build_multidoc_prompt(question, source_docs)
    answer = llm.invoke(prompt)
    return answer, source_docs

def get_multidoc_answer_with_memory(question: str, retriever, llm) -> tuple:
    if 'multi_conv_memory' not in st.session_state:
        st.session_state.multi_conv_memory = []

    memory = st.session_state.multi_conv_memory
    lang = _detect_language(question)
    
    standalone = question
    if memory:
        followup_starters = (
            'it', 'its', 'they', 'their', 'them', 'that', 'this', 'those', 'these',
            'he', 'she', 'what about', 'and', 'also', 'why', 'how about', 'what else',
            'nó', 'họ', 'còn', 'vậy', 'tại sao', 'thế', 'còn về', 'vậy thì',
            'thế còn', 'cái đó', 'điều đó',
        )
        q_lower = question.strip().lower()
        if any(q_lower.startswith(w) for w in followup_starters):
            history_lines = []
            for turn in memory[-(6):]:
                role = 'Người dùng' if turn.type == 'human' else 'Trợ lý'
                history_lines.append(f'{role}: {turn.content[:300]}')
            history_text = '\n'.join(history_lines)
            if lang == 'vi':
                condense_prompt = (
                    "Dựa vào lịch sử hội thoại, viết lại câu hỏi mới thành câu hỏi ĐỘC LẬP "
                    "đầy đủ ý nghĩa bằng TIẾNG VIỆT. CHỈ trả về câu hỏi, không giải thích.\n\n"
                    f"Lịch sử:\n{history_text}\n\nCâu hỏi mới: {question}\nCâu hỏi độc lập:"
                )
            else:
                condense_prompt = (
                    "Based on the chat history, rewrite the new question as a fully self-contained "
                    "standalone question in English. Return ONLY the rewritten question.\n\n"
                    f"History:\n{history_text}\n\nNew question: {question}\nStandalone question:"
                )
            try:
                resp = llm.invoke(condense_prompt)
                condensed = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                if condensed and len(condensed) > 5:
                    standalone = condensed
            except Exception:
                pass

    search_kwargs = getattr(retriever, 'search_kwargs', None)
    original_k = None
    if search_kwargs is not None:
        original_k = search_kwargs.get('k', RETRIEVER_K)
        search_kwargs['k'] = MULTI_DOC_RETRIEVER_K
    source_docs = retriever.invoke(standalone)

    if search_kwargs is not None and original_k is not None:
        search_kwargs['k'] = original_k

    context = _format_multidoc_chunks(source_docs)
    needs_calc = _need_calculations(question)
    cross_kw_vi = ['so sánh', 'đối chiếu', 'cả hai', 'cả ba', 'file nào', 'tài liệu nào']
    cross_kw_en = ['compare', 'both', 'which file', 'cross-reference']
    q_lower = question.lower()
    is_cross = any(kw in q_lower for kw in (cross_kw_vi + cross_kw_en))

    history_section = ''
    if memory:
        lines = []
        for turn in memory[-(6):]:
            role = ('Người dùng' if lang == 'vi' else 'User') if turn.type == 'human' else ('Trợ lý' if lang == 'vi' else 'Assistant')
            content = turn.content[:300] + ('...' if len(turn.content) > 300 else '')
            lines.append(f'{role}: {content}')
        sep = 'Lịch sử hội thoại gần nhất' if lang == 'vi' else 'Recent chat history'
        history_section = f'{sep}:\n' + '\n'.join(lines) + '\n\n'

    if lang == 'vi':
        cross_note = (
            "\n5. Câu hỏi liên quan đến NHIỀU tài liệu — phân biệt rõ thông tin từ mỗi file.\n"
            if is_cross else ""
        )
        calc_note = (
            "\n6. Yêu cầu tính toán: liệt kê số liệu → tính từng bước → kết luận.\n"
            if needs_calc else ""
        )
        prompt = (
            "Bạn là trợ lý AI chuyên phân tích đa tài liệu.\n\n"
            "═══ QUY TẮC BẮT BUỘC (VI PHẠM = SAI) ═══\n"
            "1. CHỈ dùng thông tin trong phần [NGỮ CẢNH] bên dưới.\n"
            "   Mỗi đoạn đánh nhãn: [Đoạn X | File: ... | Trang: Y].\n"
            "   KHÔNG lẫn lộn số liệu giữa các file/trang.\n"
            "   KHÔNG dùng kiến thức bên ngoài tài liệu.\n\n"
            "2. TRÍCH DẪN NGUỒN TRONG TỪNG CÂU — BẮT BUỘC:\n"
            "   Sau mỗi thông tin cụ thể, ghi ngay: (Nguồn: File [tên], Trang [số])\n"
            "   Ví dụ: 'Chỉ số X là 120 (Nguồn: File bao_cao.pdf, Trang 5).'\n"
            "   KHÔNG nêu thông tin mà thiếu trích dẫn.\n\n"
            "3. NGÔN NGỮ — QUY TẮC TUYỆT ĐỐI:\n"
            "   PHẢI viết HOÀN TOÀN bằng TIẾNG VIỆT.\n"
            "   CẤM TUYỆT ĐỐI: tiếng Trung (中文/汉字), tiếng Anh, tiếng Nga,\n"
            "   tiếng Indonesia, tiếng Nhật hay bất kỳ ngôn ngữ nào khác.\n"
            "   Ví dụ SAI: '根据文件' hoặc 'According to'\n"
            "   Ví dụ ĐÚNG: 'Theo tài liệu' hoặc 'Dựa trên tài liệu'\n\n"
            "4. Nếu không có thông tin: 'Tôi không tìm thấy thông tin này trong các tài liệu.'\n"
            "   KHÔNG bịa đặt hay suy diễn.\n"
            f"{cross_note}{calc_note}"
            "═══════════════════════════════════════════\n\n"
            f"[NGỮ CẢNH từ các tài liệu]\n{context}\n\n"
            "═══════════════════════════════════════════\n\n"
            f"{history_section}"
            f"Người dùng: {question}\n\n"
            "Trợ lý (viết hoàn toàn bằng tiếng Việt, mỗi thông tin phải có trích dẫn nguồn):"
        )
    else:
        cross_note = (
            "\n5. This question involves MULTIPLE documents — distinguish info from each file.\n"
            if is_cross else ""
        )
        calc_note = (
            "\n6. Calculation required: list figures → compute step by step → conclude.\n"
            if needs_calc else ""
        )
        prompt = (
            "You are an AI assistant specializing in multi-document analysis.\n\n"
            "═══ MANDATORY RULES (VIOLATION = INCORRECT) ═══\n"
            "1. Use ONLY information from the [CONTEXT] section below.\n"
            "   Each chunk labeled: [Chunk X | File: ... | Page: Y].\n"
            "   Do NOT mix data between files/pages.\n"
            "   Do NOT use knowledge outside the provided documents.\n\n"
            "2. INLINE CITATION IN EVERY SENTENCE — MANDATORY:\n"
            "   After each specific fact, immediately add: (Source: File [name], Page [number])\n"
            "   Example: 'Index X is 120 (Source: File report.pdf, Page 5).'\n"
            "   NEVER state information without citing the source.\n\n"
            "3. LANGUAGE — ABSOLUTE RULE:\n"
            "   Reply ENTIRELY in ENGLISH.\n"
            "   FORBIDDEN: Chinese (中文/汉字), Vietnamese, Russian, Indonesian, Japanese\n"
            "   or any other language — not even a single character.\n"
            "   WRONG: '根据文件' or 'Theo tài liệu'\n"
            "   RIGHT: 'According to the document'\n\n"
            "4. If info not found: 'I cannot find this information in the provided documents.'\n"
            "   Do NOT fabricate or infer.\n"
            f"{cross_note}{calc_note}"
            "═══════════════════════════════════════════\n\n"
            f"[CONTEXT from documents]\n{context}\n\n"
            "═══════════════════════════════════════════\n\n"
            f"{history_section}"
            f"User: {question}\n\n"
            "Assistant (English only, every fact must have inline source citation):"
        )

    answer = llm.invoke(prompt)
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)

    memory.append(HumanMessage(content=question))
    memory.append(AIMessage(content=answer_text))
    st.session_state.multi_conv_memory = memory[-6:]

    return answer, source_docs


def add_document(file_bytes: bytes, filename: str, embedder) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        if ext == 'pdf':
            loader = PDFPlumberLoader(tmp_path)
        elif ext in ('doc', 'docx'):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error(f'Unsupport file type: {ext}')
            return False

        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = st.session_state.get('chunk_size', CHUNK_SIZE),
            chunk_overlap = st.session_state.get('chunk_overlap', CHUNK_OVERLAP),
        )

        chunks = splitter.split_documents(raw_docs)

        upload_time = time.strftime("%H:%M %d/%m/%Y")
        tagged_chunks = []
        for chunk in chunks:
            new_meta = dict(chunk.metadata)
            new_meta['source_file'] = filename
            new_meta['upload_time'] = upload_time
            tagged_chunks.append(
                Document(page_content=chunk.page_content, metadata=new_meta)
            )
        
        if not tagged_chunks:
            st.error('No extractable text found.')
            return False
        
        existing = st.session_state.get('multi_vector_store')
        if existing is None:
            new_store = FAISS.from_documents(tagged_chunks, embedder)
        else:
            extra_store = FAISS.from_documents(tagged_chunks, embedder)
            existing.merge_from(extra_store)
            new_store=existing

        multi_retriever = new_store.as_retriever(
            search_type='similarity',
            search_kwargs = {'k': RETRIEVER_K},
        )

        registry = st.session_state.get('doc_registry', {})
        registry[filename] = {
            'chunks': len(tagged_chunks),
            'upload_time': upload_time,
            'ext': ext.upper(),
            'size_bytes': len(file_bytes),
        }

        st.session_state['multi_vector_store'] = new_store
        st.session_state['multi_retriever'] = multi_retriever
        st.session_state['doc_registry'] = registry

        return True
    
    except Exception as e:
        st.error(f'Failed to add document: {e}')
        return False
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def get_filtered_retirever(selected_files: list):
    store=st.session_state.get('multi_vector_store')
    if store is None:
        return st.session_state.get('retriever')
    
    if not selected_files:
        return store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': MULTI_DOC_RETRIEVER_K},
        )
    
    if len(selected_files) == 1:
        return store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {
                'k': MULTI_DOC_RETRIEVER_K,
                'filter': {'source_file': selected_files[0]},
            },
        )

    def _multi_filter(meta: dict) -> bool:
        return meta.get('source_file') in selected_files
    
    return store.as_retriever(
        search_type='similarity',
        search_kwargs = {'k': MULTI_DOC_RETRIEVER_K, 'filter': _multi_filter},
    )

get_filtered_retirever = get_filtered_retirever


def remove_document(filename: str):
    registry = st.session_state.get('doc_registry', {})
    registry.pop(filename, None)
    st.session_state['doc_registry'] = registry


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024 * 1024:
        return f'{n_bytes / 1024:.1f} KB'
    return f'{n_bytes / (1024 * 1024):.1f} MB'

def render_multi_doc_panel(embedder):
    registry = st.session_state.get('doc_registry', {})
    if not registry:
        st.subheader('Upload Documents')
    else:
        st.subheader('Add Another Document')

    uploader_key = f"multi_uploader_{st.session_state.get('multi_uploader_key', 0)}"

    new_file = st.file_uploader(
        'Drag & drop or click to browse',
        type=['pdf', 'docx', 'doc'],
        key=uploader_key,
        help='Supported: PDF, DOCX. Each file is indexed separately with its filename as metadata.'
    )

    if new_file:
        size_bytes = new_file.size
        size_display = _fmt_size(size_bytes)
        if new_file.name not in registry:
            progress_bar = st.progress(0, text='Preparing...')
            progress_bar.progress(20, text='Reading document...')
            progress_bar.progress(55, text='Splitting into chunks...')
            progress_bar.progress(75, text='Generating embeddings...')

            file_bytes = new_file.read()
            ok = add_document(file_bytes, new_file.name, embedder)

            progress_bar.progress(100, text='Done!')
            progress_bar.empty()
            if ok:
                st.session_state['doc_registry'][new_file.name]['size_display'] = size_display
                st.session_state['multi_uploader_key'] = (
                    st.session_state.get('multi_uploader_key', 0) + 1
                )
                st.session_state['active_tab'] = 'multi'
                st.rerun()
        else:
            st.info(f'**{new_file.name}** is already loaded.')

    if not registry:
        st.markdown(
            '<div style="padding:1.2rem;border:1px dashed #1e3a5f;border-radius:8px;'
            'text-align:center;color:#4a6a8a;font-size:0.85rem;margin-top:0.5rem;">'
            'No documents loaded yet.<br>'
            '<span style="font-size:0.75rem;">Upload files above to get started.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return st.session_state.get('retriever')
    
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
        'color:#0099ff;text-transform:uppercase;letter-spacing:0.08em;margin-top:1rem;">'
        'Loaded Documents</p>',
        unsafe_allow_html=True,
    )


    for fname, info in registry.items():
        size_str = info.get('size_display', _fmt_size(info.get('size_bytes', 0)))
        ext_badge = info['ext']
        chunks = info['chunks']
        upload_time = info['upload_time']
        badge_color = '#1a6b3c' if ext_badge == 'PDF' else '#1a3a6b'
        st.markdown(
            f'''<div style="
                background:#0a151f;
                border:1px solid #1e3a5f;
                border-radius:8px;
                padding:0.65rem 0.9rem;
                margin-bottom:0.45rem;
                display:flex;
                align-items:center;
                gap:0.6rem;
            ">
                <span style="
                    background:{badge_color};
                    color:#c0d8f0;
                    font-family:'IBM Plex Mono',monospace;
                    font-size:0.65rem;
                    padding:2px 7px;
                    border-radius:4px;
                    flex-shrink:0;
                ">{ext_badge}</span>
                <span style="
                    color:#c0d8f0;
                    font-size:0.82rem;
                    flex:1;
                    word-break:break-all;
                ">{fname}</span>
                <span style="
                    color:#4a6a8a;
                    font-size:0.72rem;
                    white-space:nowrap;
                    flex-shrink:0;
                ">{chunks} chunks · {size_str}</span>
            </div>''',
            unsafe_allow_html=True,
        )

    total_chunks = sum(info['chunks'] for info in registry.values()) 
    st.markdown(
        f'<div style="font-size:0.75rem;color:#4a6a8a;margin-bottom:0.6rem;">'
        f'{len(registry)} file{"s" if len(registry) > 1 else ""} &nbsp;·&nbsp; '
        f'{total_chunks} total chunks &nbsp;·&nbsp; retrieve k = {MULTI_DOC_RETRIEVER_K}'
        f'</div>',
        unsafe_allow_html=True,
    )
    
        
    all_files = list(registry.keys())
    selected = st.multiselect(
        'Filter: search only in...',
        options=all_files,
        default=[],
        placeholder='All documets (no filter)',
        key='doc_filter_select',
        help='Leave empty to search across all documents.',
    )

    if selected:
        st.markdown(
            f'<span style="font-size:0.75rem;color:#f0a000;">'
            f"Searching in: {', '.join(selected)}</span>",
            unsafe_allow_html=True,
        )

    return get_filtered_retirever(selected)