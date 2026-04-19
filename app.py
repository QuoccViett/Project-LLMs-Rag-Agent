import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from config import APP_ICON, APP_TITLE, RERANK_FERCH_K, SUPPORTED_TYPES
from utils.session import init_session
from utils.styles import inject_css
from modules.models import load_embedder, load_llm
from modules.document_processor import process_document
from modules.clear_controls import render_clear_controls
from modules.chat_history import add_to_history, render_history_sidebar
from modules.chunk_settings import render_chunk_settings
from modules.citation import render_citations, CITATION_CSS
from modules.conversational_rag import (
    get_answer_with_memory, render_memory_badge
)
from modules.qa_engine import get_answer
from modules.hybrid_search import (
    build_and_store_hybrid, render_hybrid_toggle, get_hybrid_retriever_or_fallback
)
from modules.multi_doc import render_multi_doc_panel
from modules.reranker import render_rerank_toggle, retrieve_and_rerank
from modules.self_rag import (
    self_rag_answer, render_self_rag_toggle, render_self_rag_metadata
)
from modules.qa_engine import build_prompt
from langchain_community.vectorstores import FAISS as _F 


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout='wide',
    initial_sidebar_state='expanded',
)

inject_css()
st.markdown(CITATION_CSS, unsafe_allow_html=True)
init_session()

embedder = load_embedder()
llm = load_llm()

st.session_state['embedder_ref'] = embedder

with st.sidebar:
    st.title("Management")
    st.subheader('Document Status')
    if st.session_state.doc_name:
        st.success(f'**{st.session_state.doc_name}**')
        st.markdown(f"""
        <div class="sd-stats">
            <div class="sd-stat">
                <strong>{st.session_state.doc_chunks}</strong>
                <span>Chunks</span>
            </div>
            <div class="sd-stat">
                <strong>{len(st.session_state.chat_history)}</strong>
                <span>Questions</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_memory_badge()

    else:
        st.info('No document loaded.')
    st.divider()


    render_chunk_settings()
    st.divider()

    render_hybrid_toggle()
    st.divider()

    render_rerank_toggle()
    st.divider()

    render_self_rag_toggle()
    st.divider()

    render_clear_controls()
    st.divider()

    render_history_sidebar()
    st.divider()
    st.caption('ADI v1.1')


st.markdown("""
<div class="sd-header">
    <h1>ADI: Advanced Document Intelligence</h1>
    <p>Intelligence at Your Fingertips — Ask, Discover, Learn</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_multi = st.tabs(['Single Document', 'Multi Document'])

with tab_single:
    if not st.session_state.get('doc_name'):
        st.subheader('Upload Document')
        upload_file = st.file_uploader(
            'Drag & drop or click to browse',
            type=SUPPORTED_TYPES,
            help='Supported: PDF, DOCX. Recommended size < 50MB',
            key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        )

        if upload_file:
            size_mb = upload_file.size 
            if size_mb < (1024 * 1024):
                file_size_display = f'{size_mb / 1024:.1f} KB'
            else:
                file_size_display = f'{size_mb / (1024 * 1024):.1f} MB'

            if st.session_state.doc_name != upload_file.name:
                progress_bar = st.progress(0, text='Preparing...')
                progress_bar.progress(20, text='Reading document...')
                progress_bar.progress(55, text='Splitting into chunks...')
                progress_bar.progress(75, text='Generating embeddings on GPU...')

                result = process_document(upload_file.read(), upload_file.name, embedder)

                progress_bar.progress(100, text='Done!')
                progress_bar.empty()
            
                if result:
                    result['doc_size'] = file_size_display
                    st.session_state.update(result)
                    st.session_state.conv_memory = []
                    st.session_state.chunk_metrics = []

                    if result.get('raw_docs') and result.get('vector_store'):
                        build_and_store_hybrid(result['raw_docs'], result['vector_store'])


                    st.rerun()
                else:
                    st.error('Processing failed. Please try another file.')
    else:
        f_size = st.session_state.get('doc_size', '')

        st.info(f'The system is ready with document: **{st.session_state.doc_name}** (`{f_size}`)')
        if st.button('Upload a different file'):
            st.session_state.doc_name = None
            st.session_state.doc_size = None
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
            st.rerun()
    st.divider()

    st.subheader('Ask a Question')

    use_conv = st.toggle(
        'Conversational mode (remembers previous questions)',
        value=True,
        help='When ON the assistant can answer follow-up questions using chat history.',
    )

    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None
    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = []

    if st.session_state.retriever is None:
        st.warning('Upload a document above before asking questions.')
    else:
        with st.form(key='chat_form', clear_on_submit=True):

            question = st.text_input(
                'Your question:',
                placeholder='e.g. What are the main findings? / Can you elaborate on that?',
            )
            submit_button = st.form_submit_button(label='Send')

        if submit_button and question.strip():
            with st.spinner('Search and generating answer...'):

                use_hybrid = st.session_state.get('use_hybrid', False)
                use_rerank = st.session_state.get('use_rerank', False)
                use_self_rag = st.session_state.get('use_self_rag', False)

                active_retriever = (
                    get_hybrid_retriever_or_fallback()
                    if use_hybrid
                    else st.session_state.retriever
                )

                if use_self_rag:
                    result_sr = self_rag_answer(question, active_retriever, llm)
                    st.session_state.last_self_rag_result = result_sr
                    answer = result_sr['answer']
                    sources = result_sr['source_docs']

                elif use_rerank:
                    sources = retrieve_and_rerank(
                        question, active_retriever, fetch_k=RERANK_FERCH_K
                    )
                    # context = '\n\n'.join(d.page_content for d in sources)
                    
                    answer = llm.invoke(build_prompt('', question, source_docs=sources))

                elif use_conv:
                    _orig = st.session_state.retriever
                    if use_hybrid:
                        st.session_state.retriever = active_retriever
                    answer, sources = get_answer_with_memory(
                        question, st.session_state.retriever, llm 
                    )
                    st.session_state.retriever = _orig
                else:
                    answer, sources = get_answer(
                        question, st.session_state.retriever, llm 
                    )
            
            final_answer = answer.content if hasattr(answer, 'content') else str(answer)

            add_to_history(question, final_answer)

            st.session_state.last_question = question
            st.session_state.last_answer = final_answer
            st.session_state.last_sources = sources

            st.rerun()
        
if st.session_state.last_answer:
    st.write(f'**Question:** {st.session_state.last_question}')
    st.markdown(f"""
    <div class="sd-answer">
        <div class="sd-answer-label">Answer</div>
        {st.session_state.last_answer}
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.get('use_self_rag'):
        sr_data = st.session_state.get('last_self_rag_result')
        if sr_data:
            render_self_rag_metadata(sr_data)
    render_citations(st.session_state.last_sources)
            
with tab_multi:
    st.subheader('Multi Document Q&A')
    st.caption(
        'Upload multiple document. Each is indexed separately with its filename as metadata. '
        'Use the filter to restrict to specific documents.'
    )

    active_multi_retriever = render_multi_doc_panel(embedder)

    st.divider()
    st.subheader('Ask a Question')

    if not st.session_state.get('doc_registry'):
        st.warning('Add at least one document above before asking questions.')
    else:
        q_multi = st.text_input(
            'Your question:',
            placeholder='e.g Compare the approaches described in both documents.',
            key='q_multi',
        )

        if q_multi and q_multi.strip() and active_multi_retriever:
            with st.spinner('Searching across documents...'):
                answer_m, sources_m = get_answer(q_multi, active_multi_retriever, llm)

            if hasattr(answer_m, 'content'):
                display_text = answer_m.content 
            else:
                display_text = answer_m
            st.markdown(f"""
            <div class="sd-answer">
                <div class="sd-answer-label">Answer</div>
                {display_text}
            </div>
            """, unsafe_allow_html=True)

            render_citations(sources_m)



            if sources_m:
                st.markdown('---')
                st.markdown(
                    '<p style="font-family:\'IBM Plex Mono\',monospace;'
                    'font-size:0.72rem;color:#0099ff;text-transform:uppercase;">'
                    "Sources</p>",
                    unsafe_allow_html=True,
                )
                for i, doc in enumerate(sources_m, 1):
                    src_file = doc.metadata.get('source_file', 'unknown')
                    page = doc.metadata.get('page', '')
                    page_str = f' .Page {int(page) + 1}' if page != "" else ""
                    preview = " ".join(doc.page_content.split())[:180]
                    with st.expander(f'[{i}] {src_file}{page_str} - {preview}...'):
                        st.markdown(
                            f'<div class="sd-citation-body">{doc.page_content}</div>',
                            unsafe_allow_html=True,
                        )
            add_to_history(q_multi, answer_m)