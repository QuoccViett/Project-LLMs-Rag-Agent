import os 
from re import search
import time
import tempfile
from typing import final
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


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
            search_kwargs={'k': RETRIEVER_K},
        )
    
    if len(selected_files) == 1:
        return store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {
                'k': RETRIEVER_K,
                'filter': {'source_file': selected_files[0]},
            },
        )

    def _multi_filter(meta: dict) -> bool:
        return meta.get('source_file') in selected_files
    
    return store.as_retriever(
        search_type='similarity',
        search_kwargs = {'k': RETRIEVER_K, 'filter': _multi_filter},
    )


def remove_document(filename: str):
    registry = st.session_state.get('doc_registry', {})
    registry.pop(filename, None)
    st.session_state['doc_registry'] = registry

def render_multi_doc_panel(embedder):
    st.subheader('Documents')
    registry = st.session_state.get('doc_registry', {})

    new_file = st.file_uploader(
        'Add another document',
        type=['pdf', 'docx', 'doc'],
        key='multi_uploader',
        help='Each uploaded document is added to the shared index.'
    )

    if new_file:
        if new_file.name not in registry:
            with st.spinner(f'Adding {new_file.name}...'):
                file_bytes = new_file.read()
                ok = add_document(file_bytes, new_file.name, embedder)
            if ok:
                st.success(f'Added: {new_file.name}')
                st.rerun()

    if not registry:
        st.caption('No documents loaded.')
        return st.session_state.get('retriever')
    
    st.markdown('**Loaded documents:**')
    for fname, info in registry.items():
        col_name, col_info = st.columns([3, 2])
        col_name.markdown(f'{fname}')
        col_info.caption(f"{info['ext']} . {info['chunks']} chunks . {info['upload_time']}")
        
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