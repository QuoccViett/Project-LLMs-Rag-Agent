import os
import json
import hashlib
import tempfile
import time
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, EMBEDDING_MODEL


_ADI_CACHE_DIR = os.getenv('ADI_CACHE_DIR')


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _default_user_cache_base() -> str:
    """Return a user-writable cache base directory (OS-specific).

    Windows: %LOCALAPPDATA%\\Project-LLMs-Rag-Agent\\adi_cache
    Linux/macOS: $XDG_CACHE_HOME or ~/.cache/Project-LLMs-Rag-Agent/adi_cache
    """
    local_app_data = os.getenv('LOCALAPPDATA')
    if local_app_data:
        return os.path.join(local_app_data, 'Project-LLMs-Rag-Agent', 'adi_cache')

    xdg_cache = os.getenv('XDG_CACHE_HOME')
    if xdg_cache:
        return os.path.join(xdg_cache, 'Project-LLMs-Rag-Agent', 'adi_cache')

    home = os.path.expanduser('~')
    return os.path.join(home, '.cache', 'Project-LLMs-Rag-Agent', 'adi_cache')


def _cache_base() -> str:
    # If ADI_CACHE_DIR is provided, allow absolute paths.
    # If it's relative, keep backward compatibility by resolving under project root.
    if _ADI_CACHE_DIR:
        if os.path.isabs(_ADI_CACHE_DIR):
            return _ADI_CACHE_DIR
        return os.path.join(_project_root(), _ADI_CACHE_DIR)
    return _default_user_cache_base()


def _cache_root() -> str:
    return os.path.join(_cache_base(), 'single')


def _safe_filename(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.', ' ') else '_' for c in (name or 'document'))


def _cache_key(file_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(b'\n')
    h.update((filename or '').encode('utf-8', errors='ignore'))
    h.update(b'\n')
    h.update(str(chunk_size).encode('ascii'))
    h.update(b':')
    h.update(str(chunk_overlap).encode('ascii'))
    h.update(b'\n')
    h.update((EMBEDDING_MODEL or '').encode('utf-8', errors='ignore'))
    return h.hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(_cache_root(), key)


def _meta_path(key: str) -> str:
    return os.path.join(_cache_path(key), 'meta.json')


def _coerce_to_document(obj) -> Document | None:
    if obj is None:
        return None
    if isinstance(obj, Document):
        return obj
    if isinstance(obj, str):
        return Document(page_content=obj, metadata={})
    if isinstance(obj, dict):
        page_content = obj.get('page_content') or obj.get('content') or obj.get('text')
        if page_content is None:
            return None
        meta = obj.get('metadata') if isinstance(obj.get('metadata'), dict) else {}
        return Document(page_content=str(page_content), metadata=meta)
    # Fallback: best-effort string conversion.
    try:
        return Document(page_content=str(obj), metadata={})
    except Exception:
        return None


def _extract_docs_from_faiss(vector_store: FAISS) -> list[Document]:
    ids = getattr(vector_store, 'index_to_docstore_id', None)
    docstore = getattr(vector_store, 'docstore', None)
    if not ids or docstore is None:
        return []
    out = []
    for doc_id in ids:
        try:
            doc = docstore.search(doc_id)
            coerced = _coerce_to_document(doc)
            if coerced is not None:
                out.append(coerced)
        except Exception:
            continue
    return out


def _try_load_cached_faiss(key: str, embedder):
    path = _cache_path(key)
    if not os.path.isdir(path):
        return None
    try:
        # Newer LangChain requires allow_dangerous_deserialization for local pickles.
        return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    except TypeError:
        try:
            return FAISS.load_local(path, embedder)
        except Exception:
            return None
    except Exception:
        return None


def _save_cached_faiss(key: str, vector_store: FAISS, meta: dict):
    path = _cache_path(key)
    try:
        os.makedirs(path, exist_ok=True)
        vector_store.save_local(path)
        with open(_meta_path(key), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        # Cache failures should never break the app.
        pass

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
        'documents': documents,
    }

def process_document(file_bytes: bytes, filename: str, embedder):
    ext = filename.rsplit('.', 1)[-1].lower()
    tmp_path = None
    chunk_size = int(st.session_state.get('chunk_size', CHUNK_SIZE) or CHUNK_SIZE)
    chunk_overlap = int(st.session_state.get('chunk_overlap', CHUNK_OVERLAP) or CHUNK_OVERLAP)
    key = _cache_key(file_bytes, filename, chunk_size, chunk_overlap)

    cached_store = _try_load_cached_faiss(key, embedder)
    if cached_store is not None:
        docs = _extract_docs_from_faiss(cached_store)
        # Ensure citations show the original uploaded filename (not a temp path).
        for d in docs:
            meta = getattr(d, 'metadata', None)
            if isinstance(meta, dict):
                meta['source_file'] = filename
        retriever = cached_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': RETRIEVER_K},
        )
        avg_len = int(sum(len(d.page_content) for d in docs) / len(docs)) if docs else 0
        return {
            'vector_store': cached_store,
            'retriever': retriever,
            'doc_chunks': len(docs),
            'avg_chunk_len': avg_len,
            'doc_name': filename,
            'doc_bytes': file_bytes,
            'raw_docs': docs,  # chunk docs from cache; enough for hybrid BM25
        }

    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = _get_loader(tmp_path, ext)
        raw_docs = loader.load()

        # Replace temp file path with the original uploaded filename for UI citations.
        for d in raw_docs:
            meta = getattr(d, 'metadata', None)
            if isinstance(meta, dict):
                meta['source_file'] = filename

        result = _build_index(raw_docs, embedder, chunk_size, chunk_overlap)
        if result is None:
            return None
        
        result['doc_name'] = filename
        result['doc_bytes'] = file_bytes
        # Keep raw docs for hybrid BM25 and for potential re-index.
        # (Also store chunk docs under result['documents'].)
        result['raw_docs'] = raw_docs

        meta = {
            'filename': _safe_filename(filename),
            'created_at': int(time.time()),
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'embedding_model': EMBEDDING_MODEL,
            'doc_chunks': result.get('doc_chunks', 0),
        }
        _save_cached_faiss(key, result['vector_store'], meta)
        return result
    
    except Exception as e:
        st.error(f'Failed to process document: {e}')
        return None
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def rebuild_index(file_bytes: bytes, filename: str, embedder, chunk_size: int, chunk_overlap: int):
    key = _cache_key(file_bytes, filename, int(chunk_size), int(chunk_overlap))
    cached_store = _try_load_cached_faiss(key, embedder)
    if cached_store is not None:
        docs = _extract_docs_from_faiss(cached_store)
        for d in docs:
            meta = getattr(d, 'metadata', None)
            if isinstance(meta, dict):
                meta['source_file'] = filename
        retriever = cached_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': RETRIEVER_K},
        )
        avg_len = int(sum(len(d.page_content) for d in docs) / len(docs)) if docs else 0
        return {
            'vector_store': cached_store,
            'retriever': retriever,
            'doc_chunks': len(docs),
            'avg_chunk_len': avg_len,
            'doc_name': filename,
            'raw_docs': docs,
            'documents': docs,
        }

    raw_docs = st.session_state.get('raw_docs')
    if raw_docs is None:
        ext = filename.rsplit('.', 1)[-1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            raw_docs = _get_loader(tmp_path, ext).load()
            for d in raw_docs:
                meta = getattr(d, 'metadata', None)
                if isinstance(meta, dict):
                    meta['source_file'] = filename
        except Exception as e:
            st.error(f'Re-index failed: {e}')
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    result = _build_index(raw_docs, embedder, chunk_size, chunk_overlap)
    if result:
        result['doc_name'] = filename
        meta = {
            'filename': _safe_filename(filename),
            'created_at': int(time.time()),
            'chunk_size': int(chunk_size),
            'chunk_overlap': int(chunk_overlap),
            'embedding_model': EMBEDDING_MODEL,
            'doc_chunks': result.get('doc_chunks', 0),
        }
        _save_cached_faiss(key, result['vector_store'], meta)
    return result
