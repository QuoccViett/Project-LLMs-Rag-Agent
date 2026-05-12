import os
import json
import hashlib
import io
import tempfile
import time
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, EMBEDDING_MODEL
import re
import pdfplumber


_ADI_CACHE_DIR = os.getenv('ADI_CACHE_DIR')

# Bump this when changing extraction/chunking semantics so cached indexes are rebuilt.
_INDEX_SCHEMA_VERSION = 'structuredpdf_v3'


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
    h.update(_INDEX_SCHEMA_VERSION.encode('ascii'))
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
    if ext in ('docx', 'doc'):
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f'Unsupported file type: {ext}')


def _table_to_markdown(table: list[list]) -> str:
    if not table:
        return ''

    rows = []
    for row in table:
        if row is None:
            continue
        cleaned = [str(c).strip() if c is not None else '' for c in row]
        rows.append(cleaned)
    if not rows:
        return ''

    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 0:
        return ''

    def _pad(r: list[str]) -> list[str]:
        r = list(r)
        if len(r) < max_cols:
            r.extend([''] * (max_cols - len(r)))
        return r

    def _cell(x: str) -> str:
        return (x or '').replace('|', '\\|').replace('\n', ' ').strip()

    rows = [_pad(r) for r in rows]
    header = rows[0]
    body = rows[1:]

    header_line = '| ' + ' | '.join(_cell(c) for c in header) + ' |'
    sep_line = '| ' + ' | '.join(['---'] * max_cols) + ' |'
    body_lines = ['| ' + ' | '.join(_cell(c) for c in r) + ' |' for r in body]
    return '\n'.join([header_line, sep_line] + body_lines)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: (t[0], t[1]))
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_bbox_text_excluding_tables(page, bbox: tuple[float, float, float, float], table_bboxes: list[tuple[float, float, float, float]]) -> str:
    """Extract text from bbox while skipping regions that overlap with tables.

    Strategy: compute vertical y-intervals within bbox not covered by any
    intersecting table bbox (within the same x-range), then extract text from
    those slices and join.
    """
    x0, top, x1, bottom = bbox
    if bottom <= top or x1 <= x0:
        return ''

    # Collect y-intervals (top,bottom) of tables intersecting this bbox.
    blocked: list[tuple[float, float]] = []
    for tx0, ttop, tx1, tbottom in (table_bboxes or []):
        # Check x-overlap first.
        if tx1 <= x0 or tx0 >= x1:
            continue
        # Clamp to bbox y-range.
        y0 = max(float(ttop), float(top))
        y1 = min(float(tbottom), float(bottom))
        if y1 > y0:
            blocked.append((y0, y1))

    blocked = _merge_intervals(blocked)
    if not blocked:
        return (page.within_bbox(bbox).extract_text(layout=True) or '').strip()

    parts: list[str] = []
    cursor = float(top)
    for y0, y1 in blocked:
        if y0 > cursor:
            slice_bbox = (x0, cursor, x1, y0)
            txt = (page.within_bbox(slice_bbox).extract_text(layout=True) or '').strip()
            if txt:
                parts.append(txt)
        cursor = max(cursor, float(y1))

    if cursor < float(bottom):
        slice_bbox = (x0, cursor, x1, bottom)
        txt = (page.within_bbox(slice_bbox).extract_text(layout=True) or '').strip()
        if txt:
            parts.append(txt)

    return '\n'.join(parts).strip()


def _extract_structured_pdf_docs(pdf, source_filename: str) -> list[Document]:
    """Extract PDF content with better structure:

    - Tables first (as Markdown)
    - Then text by columns (left then right)

    This helps multi-column papers and table queries like "Table 1".
    """
    docs: list[Document] = []
    text_parts: list[str] = []
    for page_idx, page in enumerate(pdf.pages):
        page_text_full = page.extract_text() or ''
        caption_ids = re.findall(r'\b(?:Table|Bảng)\s*\d+(?:\.\d+)?\b', page_text_full, flags=re.IGNORECASE)

        tables = page.extract_tables() or []
        for t_idx, table in enumerate(tables):
            table_id = caption_ids[t_idx] if t_idx < len(caption_ids) else f'Table {page_idx + 1}.{t_idx + 1}'
            md = _table_to_markdown(table)
            if not md.strip():
                continue
            content = (
                f"[{table_id} | Page {page_idx + 1}]\n"
                f"{md}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'page': page_idx,
                        'source_file': source_filename,
                        'content_type': 'table',
                        'table_id': table_id,
                        'table_index': t_idx,
                    },
                )
            )

        table_bboxes: list[tuple[float, float, float, float]] = []
        try:
            table_objects = page.find_tables() or []
            for t in table_objects:
                bb = getattr(t, 'bbox', None)
                if bb and len(bb) == 4:
                    table_bboxes.append(tuple(float(v) for v in bb))
        except Exception:
            table_bboxes = []

        width = float(getattr(page, 'width', 0) or 0)
        height = float(getattr(page, 'height', 0) or 0)
        if width > 0 and height > 0:
            left_bbox = (0, 0, width / 2, height)
            right_bbox = (width / 2, 0, width, height)

            left_text = _extract_bbox_text_excluding_tables(page, left_bbox, table_bboxes)
            right_text = _extract_bbox_text_excluding_tables(page, right_bbox, table_bboxes)

            combined = (
                f"--- Trang {page.page_number} ---\n"
                f"\n[VĂN BẢN CỘT TRÁI]\n{left_text}\n"
                f"\n[VĂN BẢN CỘT PHẢI]\n{right_text}\n"
            ).strip()
        else:
            combined = (page_text_full or '').strip()

        if combined:
            # Insert a marker so we can recover page metadata after chunking.
            # We strip these markers back out during chunk post-processing.
            text_parts.append(f"\n\n<ADI_PAGE:{page_idx + 1}>\n\n{combined}")

    full_text = ''.join(text_parts).strip()
    if full_text:
        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    'source_file': source_filename,
                    'content_type': 'text',
                    'layout': 'two_column',
                },
            )
        )

    return docs


def extract_structured_pdf_docs(pdf_path: str, source_filename: str) -> list[Document]:
    """Path-based wrapper for structured PDF extraction."""
    with pdfplumber.open(pdf_path) as pdf:
        return _extract_structured_pdf_docs(pdf, source_filename)


def extract_structured_pdf_docs_from_bytes(file_bytes: bytes, source_filename: str) -> list[Document]:
    """Bytes-based extraction (used by process_document to avoid temp files for PDFs)."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return _extract_structured_pdf_docs(pdf, source_filename)


def _load_raw_docs(tmp_path: str, ext: str, source_filename: str) -> list[Document]:
    if ext == 'pdf':
        return extract_structured_pdf_docs(tmp_path, source_filename)
    loader = _get_loader(tmp_path, ext)
    raw_docs = loader.load()
    for d in raw_docs:
        meta = getattr(d, 'metadata', None)
        if isinstance(meta, dict):
            meta['source_file'] = source_filename
    return raw_docs
    

def _build_index(raw_docs: list, embedder, chunk_size: int, chunk_overlap: int):
    # Prefer larger chunks for table markdown so a whole table
    # is more likely to stay within a single retrieved context.
    chunk_size_table = max(int(chunk_size), 4000)
    chunk_overlap_table = max(int(chunk_overlap), 200)

    # Prefer splitting by paragraph boundaries to preserve meaning.
    separators = ["\n\n", "\n", ". ", " ", ""]
    splitter_text = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    splitter_table = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_table,
        chunk_overlap=chunk_overlap_table,
        separators=separators,
    )

    raw_docs = raw_docs or []
    table_docs = [d for d in raw_docs if getattr(d, 'metadata', {}).get('content_type') == 'table']
    text_docs = [d for d in raw_docs if d not in table_docs]

    documents = []
    if text_docs:
        text_chunks = splitter_text.split_documents(text_docs)

        # Recover page numbers from markers inserted during PDF extraction.
        page_re = re.compile(r"<ADI_PAGE:(\d+)>")
        current_page = None
        for ch in text_chunks:
            content = ch.page_content or ''
            markers = page_re.findall(content)
            if markers:
                current_page = int(markers[-1]) - 1
            meta = getattr(ch, 'metadata', None)
            if isinstance(meta, dict) and current_page is not None and 'page' not in meta:
                meta['page'] = current_page
            ch.page_content = page_re.sub('', content).strip()

        documents.extend(text_chunks)
    if table_docs:
        documents.extend(splitter_table.split_documents(table_docs))
    for d in documents:
        meta = getattr(d, 'metadata', None)
        if isinstance(meta, dict):
            meta['adi_is_chunk'] = True

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
                meta['adi_is_chunk'] = True
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
        # PDFs: use pdfplumber directly to prevent cross-column reading issues.
        # Other file types: keep the existing temp-file loader flow.
        if ext == 'pdf':
            raw_docs = extract_structured_pdf_docs_from_bytes(file_bytes, filename)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            raw_docs = _load_raw_docs(tmp_path, ext, filename)

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
    # If session raw_docs are already chunks (e.g. loaded from cache),
    # rebuild from original bytes to avoid double-splitting.
    if raw_docs and any(getattr(d, 'metadata', {}).get('adi_is_chunk') for d in raw_docs if hasattr(d, 'metadata')):
        raw_docs = None
    if raw_docs is None:
        ext = filename.rsplit('.', 1)[-1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            raw_docs = _load_raw_docs(tmp_path, ext, filename)
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
