import streamlit as st 

def _get_page(doc) -> str:
    meta = getattr(doc, 'metadata', {}) or {}
    if 'page' in meta:
        return str(int(meta['page']) +1)
    if 'page_number' in meta:
        return str(meta['page_number'])
    return '-'

def _make_preview(text: str, max_chars: int = 200) -> str:
    single = ' '.join(text.split())
    if len(single) <= max_chars:
        return single
    return single[:max_chars].rsplit(' ', 1)[0] + '...'

def render_citations(source_docs: list):
    if not source_docs:
        return
    
    st.markdown('---')
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;'
        'font-size:0.72rem;color:#0099ff;'
        'text-transform:uppercase;letter-spacing:0.08em;">'
        "Sources</p>",
        unsafe_allow_html=True,
    )

    for i, doc in enumerate(source_docs, 1):
        page = _get_page(doc)
        preview = _make_preview(doc.page_content, max_chars=200)
        label = f'[{i}] Page {page} - {preview}'

        with st.expander(label, expanded=False):
            st.markdown(
                f'<div class="sd-citation-header">'
                f'Citation [{i}] · Page {page}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="sd-citation-body">{doc.page_content}</div>',
                unsafe_allow_html=True,
            )

def build_answer_with_inline_refs(answer: str, source_docs: list) -> str:
    refs = ' ' + ''.join(f'[{i}]' for i in range(1, len(source_docs) + 1))
    return answer.rstrip() + refs


CITATION_CSS = '''
<style>
.sd-citation-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #0099ff;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.sd-citation-body {
    background: #090f16;
    border: 1px solid #1a2e42;
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    font-size: 0.83rem;
    color: #7aaed0;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
    max-height: 320px;
    overflow-y: auto;
}
</style>
'''