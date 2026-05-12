# document_processor.py
import os
import tempfile
import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K

# ---------------------------------------------------------------------------
# pdfplumber tolerance constants
# x_tolerance=2 is critical: x_tolerance=3 causes Vietnamese characters that
# sit close together to be merged without spaces, producing garbled text like
# "GiaodiệnStreamlit" instead of "Giao diện Streamlit".
# ---------------------------------------------------------------------------
_X_TOL = 2
_Y_TOL = 2


def _find_page_header_bottom(page) -> float:
    """
    Tìm y-coordinate cuối cùng của page header (running header ở đầu trang).
    Page header IEEE thường là 1 dòng text full-width ở ~y=35-50.
    Trả về y để crop từ đó trở xuống là body.
    """
    words = page.extract_words(x_tolerance=_X_TOL, y_tolerance=_Y_TOL)
    if not words:
        return 0.0

    sorted_tops = sorted(set(round(float(w['top'])) for w in words))
    if not sorted_tops:
        return 0.0

    first_y = sorted_tops[0]
    first_line = [w for w in words if abs(float(w['top']) - first_y) < 8]

    page_w = page.width
    has_cross = any(
        float(w['x0']) < page_w * 0.2 and float(w['x1']) > page_w * 0.8
        for w in first_line
    )

    if has_cross:
        header_bottom = max(float(w['bottom']) for w in first_line)
        return header_bottom + 5.0

    return 0.0


def _detect_fullwidth_bottom(page, body_top: float, mid_x: float) -> float:
    """
    Từ body_top trở xuống, tìm y bắt đầu của vùng 2 cột thực sự.
    Full-width zone = title, abstract, keywords (text kéo dài qua mid_x).
    Trả về y bắt đầu 2 cột. Nếu không có full-width → trả về body_top.
    """
    words = page.extract_words(x_tolerance=_X_TOL, y_tolerance=_Y_TOL)
    body_words = [w for w in words if float(w['top']) >= body_top]
    if not body_words:
        return body_top

    lines: dict = {}
    for w in body_words:
        y = round(float(w['top']) / 6) * 6
        lines.setdefault(y, []).append(w)

    streak = 0
    candidate_y = None

    for y in sorted(lines):
        lw = lines[y]
        has_left  = any(float(w['x1']) <= mid_x - 10 for w in lw)
        has_right = any(float(w['x0']) >= mid_x + 10 for w in lw)
        has_cross = any(
            float(w['x0']) < mid_x - 10 and float(w['x1']) > mid_x + 10
            for w in lw
        )

        if has_left and has_right and not has_cross:
            streak += 1
            if streak == 1:
                candidate_y = y
            if streak >= 2:
                return max(body_top, candidate_y - 2)
        else:
            streak = 0
            candidate_y = None

    return body_top


def _extract_pdf_column_aware(file_path: str) -> list[Document]:
    """
    Extract PDF có xử lý layout 2 cột (IEEE/academic paper style).

    Fix: uses x_tolerance=2, y_tolerance=2 (down from 3) so pdfplumber
    does NOT merge adjacent Vietnamese glyphs into space-less blobs.

    Pipeline per page:
      1. Detect running page header → crop away
      2. Detect full-width zone (title, abstract)
      3. Detect two-column zone
      4. Extract left column then right column separately
    """
    docs = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            w, h = page.width, page.height
            mid_x = w / 2

            words = page.extract_words(x_tolerance=_X_TOL, y_tolerance=_Y_TOL)
            if not words:
                continue

            # --- Step 1: find body top (skip running header) ---
            header_bottom = _find_page_header_bottom(page)

            # --- Step 2: check two-column layout ---
            body_ws = [ww for ww in words if float(ww['top']) > header_bottom + 5]
            left_cnt  = sum(1 for ww in body_ws if float(ww['x1']) < mid_x - 20)
            right_cnt = sum(1 for ww in body_ws if float(ww['x0']) > mid_x + 20)
            is_two_col = left_cnt > 10 and right_cnt > 10

            if not is_two_col:
                # Single-column page
                text = ''
                if header_bottom > 2:
                    hw = page.crop((0, 0, w, header_bottom))
                    text = (hw.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                body = page.crop((0, max(0, header_bottom), w, h))
                body_text = (body.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                full_text = '\n\n'.join(p for p in [text, body_text] if p)
            else:
                # --- Step 3: find full-width zone (title/abstract) ---
                two_col_y = _detect_fullwidth_bottom(page, header_bottom, mid_x)

                parts: list[str] = []

                # Page header
                if header_bottom > 2:
                    hdr = page.crop((0, 0, w, header_bottom))
                    hdr_text = (hdr.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                    if hdr_text:
                        parts.append(hdr_text)

                # Full-width zone (title, abstract, keywords)
                if two_col_y > header_bottom + 5:
                    fw = page.crop((0, header_bottom, w, two_col_y))
                    fw_text = (fw.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                    if fw_text:
                        parts.append(fw_text)

                # Left column (read first)
                lc = page.crop((0, two_col_y, mid_x - 5, h))
                left_text = (lc.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                if left_text:
                    parts.append(left_text)

                # Right column (read second)
                rc = page.crop((mid_x + 5, two_col_y, w, h))
                right_text = (rc.extract_text(x_tolerance=_X_TOL, y_tolerance=_Y_TOL) or '').strip()
                if right_text:
                    parts.append(right_text)

                full_text = '\n\n'.join(parts)

            if full_text.strip():
                docs.append(Document(
                    page_content=full_text.strip(),
                    metadata={
                        'source':      os.path.basename(file_path),
                        'source_file': os.path.basename(file_path),
                        'page':        page_num,
                    }
                ))

    return docs


def _get_loader(file_path: str, ext: str):
    if ext in ('docx', 'doc'):
        return Docx2txtLoader(file_path)
    raise ValueError(f'Unsupported file type: {ext}')


def _build_index(raw_docs: list, embedder, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.split_documents(raw_docs)

    if not documents:
        st.error('No extractable text found in this document.')
        return None

    vector_store = FAISS.from_documents(documents, embedder)
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': RETRIEVER_K},
    )
    avg_len = int(sum(len(d.page_content) for d in documents) / len(documents))

    return {
        'vector_store': vector_store,
        'retriever':    retriever,
        'doc_chunks':   len(documents),
        'avg_chunk_len': avg_len,
    }


def process_document(file_bytes: bytes, filename: str, embedder):
    ext = filename.rsplit('.', 1)[-1].lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if ext == 'pdf':
            raw_docs = _extract_pdf_column_aware(tmp_path)
        else:
            raw_docs = _get_loader(tmp_path, ext).load()

        if not raw_docs:
            st.error('No text extracted from document.')
            return None

        result = _build_index(raw_docs, embedder, CHUNK_SIZE, CHUNK_OVERLAP)
        if result is None:
            return None

        result['doc_name']  = filename
        result['doc_bytes'] = file_bytes
        result['raw_docs']  = raw_docs
        return result

    except Exception as e:
        st.error(f'Failed to process document: {e}')
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def rebuild_index(file_bytes: bytes, filename: str, embedder,
                  chunk_size: int, chunk_overlap: int):
    raw_docs = st.session_state.get('raw_docs')
    ext = filename.rsplit('.', 1)[-1].lower()

    if raw_docs is None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            if ext == 'pdf':
                raw_docs = _extract_pdf_column_aware(tmp_path)
            else:
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