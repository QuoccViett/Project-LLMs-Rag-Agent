import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K

def _get_loader(file_path: str, ext: str):
    if ext == 'pdf':
        return PDFPlumberLoader(file_path)
    elif ext in ('docx', 'doc'):
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    

def _build_index(raw_docs: list, embedder, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    documents = splitter.split_documents(raw_docs)

    if not documents:
        st.error('No extractable text found in this document.')
        return None
    
    vector_store = FAISS.from_documents(documents, embedder)
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': RETRIEVER_K}
    )
    avg_len = int(sum(len(d.page_content) for d in documents) / len(documents))

    return {
        'vector_store': vector_store,
        'retriever': retriever,
        'doc_chunks': len(documents),
        'avg_chunk_len': avg_len,
    }

def process_document(file_bytes: bytes, filename: str, embedder):
    ext = filename.rsplit('.', 1)[-1].lower()
    tmp_path = None

    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = _get_loader(tmp_path, ext)
        raw_docs = loader.load()

        result = _build_index(raw_docs, embedder, CHUNK_SIZE, CHUNK_OVERLAP)
        if result is None:
            return None
        
        result['doc_name'] = filename
        result['doc_bytes'] = file_bytes
        result['raw_docs'] = raw_docs
        return result
    
    except Exception as e:
        st.error(f'Failed to process document: {e}')
        return None
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def rebuild_index(file_bytes: bytes, filename: str, embedder, chunk_size: int, chunk_overlap: int):
    raw_docs = st.session_state.get('raw_docs')
    if raw_docs is None:
        ext = filename.rsplit('.', 1)[-1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            raw_docs = _get_loader(tmp_path, ext).load()
        except Exception as e:
            st.error(f'Re-index failed: {e}')
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    result = _build_index(raw_docs, embedder, chunk_size, chunk_overlap)
    if result:
        result['doc_name'] = filename
    return result
