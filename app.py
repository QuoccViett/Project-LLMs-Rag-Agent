import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from config import APP_ICON, APP_TITLE, SUPPORTED_TYPES
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

    render_clear_controls()

    st.divider()

    render_history_sidebar()
    st.divider()
    st.caption('SmartDoc AI v1.1')


st.markdown("""
<div class="sd-header">
    <h1>SmartDoc AI</h1>
    <p>Intelligent Document Q&amp;A — ask anything about your documents</p>
    <span class="sd-badge">Qwen2.5:3b</span>
    <span class="sd-badge">MPNet 768-dim</span>
    <span class="sd-badge">FAISS</span>
    <span class="sd-badge">Conv. Memory</span>
</div>
""", unsafe_allow_html=True)


st.subheader('Upload Document')

upload_file = st.file_uploader(
    'Drag & drop or click to browse',
    type=SUPPORTED_TYPES,
    help='Supported: PDF, DOCX. Recommended size < 50MB',
    key=f"uploader_{st.session_state.get('uploader_key', 0)}",
)

if upload_file:
    size_mb = upload_file.size / ( 1024 * 1024 )

    st.text(f'File: {upload_file.name} ({size_mb:.1f} MB)')

    if st.session_state.doc_name != upload_file.name:
        progress_bar = st.progress(0, text='Preparing...')
        progress_bar.progress(20, text='Reading document...')
        progress_bar.progress(55, text='Splitting into chunks...')
        progress_bar.progress(75, text='Generating embeddings on GPU...')

        result = process_document(upload_file.read(), upload_file.name, embedder)

        progress_bar.progress(100, text='Done!')
        progress_bar.empty()
    
        if result:
            st.session_state.update(result)
            st.session_state.conv_memory = []
            st.session_state.chunk_metrics = []
            st.success(
                f"Ready — **{result['doc_chunks']} chunks** "
                f"(avg {result['avg_chunk_len']} chars) indexed from "
                f"**{upload_file.name}**"
            )
        else:
            st.error('Processing failed. Please try another file.')
st.divider()


st.subheader('Ask a Question')

use_conv = st.toggle(
    'Conversational mode (remembers previous questions)',
    value=True,
    help='When ON the assistant can answer follow-up questions using chat history.',
)

if st.session_state.retriever is None:
    st.warning('Upload a document above before asking questions.')
else:
    question = st.text_input(
        'Your question:',
        placeholder='e.g. What are the main findings? / Can you elaborate on that?',
    )

    if question and question.strip():
        with st.spinner('Search and generating answer...'):
            if use_conv:
                answer, sources = get_answer_with_memory(
                    question, st.session_state.retriever, llm 
                )
            else:
                answer, sources = get_answer(
                    question, st.session_state.retriever, llm 
                )
        
        st.markdown(f"""
        <div class="sd-answer">
            <div class="sd-answer-label">Answer</div>
            {answer}
        </div>
        """, unsafe_allow_html=True)

        render_citations(sources)

        add_to_history(question, answer)