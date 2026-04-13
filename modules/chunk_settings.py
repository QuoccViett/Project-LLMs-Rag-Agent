import time
from unittest import result
import streamlit as st 
from config import CHUNK_SIZE_OPTIONS, CHUNK_OVERLAP_OPTIONS
from modules.document_processor import rebuild_index

def _apply_chunk_strategy(size: int, overlap: int):
    raw_bytes = st.session_state.get('doc_bytes')
    filename = st.session_state.doc_name
    if not raw_bytes:
        st.warning('Original file bytes not cached - re-upload to change chunk settings.')
        return
    
    embedder = st.session_state.get('embedder_ref')
    if embedder is None:
        st.error('Embedder not available.')
        return
    
    with st.spinner(f'Re-indexing with size={size}, overlap={overlap}...'):
        t0 = time.perf_counter()
        result = rebuild_index(raw_bytes, filename, embedder, size, overlap)
        elapsed = time.perf_counter() - t0

    if result:
        st.session_state.update(result)
        st.session_state.chunk_size = size
        st.session_state.chunk_overlap = overlap
        st.session_state.conv_memory = []
        st.session_state.chunk_metrics.append({
            'size': size,
            'overlap': overlap,
            'chunks': result['doc_chunks'],
            'avg_len': result['avg_chunk_len'],
            'index_s': f'{elapsed:.1f}s',
        })
        st.success(f"Re-indexed: **{result['doc_chunk']} chunks** in {elapsed:.1f}s")
        st.rerun()

def _render_metrics_table():
    rows = ''
    for i, m in enumerate(st.session_state.chunk_metrics):
        active = (
            m['size'] == st.session_state.chunk_size
            and m['overlap'] == st.session_state.chunk_overlap
        )
        style = 'background:#0d2137;font-weight:600;' if active else ''
        row += (
            f"<tr style='{style}'>"
            f"<td>#{i+1}</td>"
            f"<td>{m['size']}</td>"
            f"<td>{m['overlap']}</td>"
            f"<td>{m['chunks']}</td>"
            f"<td>{m['avg_len']}</td>"
            f"<td>{m['index_s']}</td>"
            f"</tr>"
        )
    st.markdown(f'''
    <table style="width:100%;font-size:0.75rem;border-collapse:collapse;color:#c0d8f0;">
      <thead>
        <tr style="color:#0099ff;border-bottom:1px solid #1e3a5f;">
          <th>#</th><th>Size</th><th>Overlap</th>
          <th>Chunks</th><th>Avg len</th><th>Time</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    ''', unsafe_allow_html=True)

def render_chunk_settings():
    st.subheader('Chunk Strategy')
    new_size = st.selectbox(
        "Chunk size (chars)",
        options=CHUNK_SIZE_OPTIONS,
        index=CHUNK_SIZE_OPTIONS.index(st.session_state.chunk_size),
        key='sel_chunk_size',
        help='Large chunks = more context per retrieval, but heavier on VRAM.'
    )
    new_overlap = st.selectbox(
        'Chunk overlap (chars)',
        options=CHUNK_OVERLAP_OPTIONS,
        index=CHUNK_OVERLAP_OPTIONS.index(st.session_state.chunk_overlap),
        key='sel_chunk_overlap',
        help='Overlap prevents context from being lost at chunk boundaries.'
    )
    apply_disabled = (
        st.session_state.vector_store is None
        or (
            new_size == st.session_state.chunk_size
            and new_overlap == st.session_state.chunk_overlap
        )
    )
    if st.button('Apply & Re-index', use_container_width=True,
                 disabled=apply_disabled, key='btn_apply_chunk'):
        _apply_chunk_strategy(new_size, new_overlap)

    if st.session_state.chunk_metrics:
        st.markdown('Strategy comparison')
        _render_metrics_table()