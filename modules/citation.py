from ast import pattern
import re
from sys import prefix
import streamlit as st 
from config  import CITATION_TOP_N, CITATION_PREVIEW_CHARS, CITATION_HIGHLIGHT_CHARS


def _get_source_file(doc) -> str:
    meta = getattr(doc, 'metadata', {}) or {}
    return meta.get('source_file', meta.get('source', ''))

def _get_page(doc) -> str:
    meta = getattr(doc, 'metadata', {}) or {}
    if 'page' in meta:
        return str(int(meta['page']) +1)
    if 'page_number' in meta:
        return str(meta['page_number'])
    return '-'

def _make_preview(text: str, max_chars: int = CITATION_PREVIEW_CHARS) -> str:
    single = ' '.join(text.split())
    if len(single) <= max_chars:
        return single
    return single[:max_chars].rsplit(' ', 1)[0] + '...'

def _score_doc(doc, question: str) -> float:
    if not question:
        return 0.0
    q_tokens = set(re.findall(r'\w+', question.lower()))
    d_tokens = set(re.findall(r'\w+', doc.page_content.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)

def _dedup_docs(source_docs: list) -> list:
    seen = set()
    out = []
    for doc in source_docs:
        key = doc.page_content[:80].strip()
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out

def _select_top_docs(source_docs: list, question: str, top_n: int = CITATION_TOP_N) -> list:
    deduped = _dedup_docs(source_docs)
    if not question:
        return deduped[:top_n]
    scored = sorted(deduped, key=lambda d: _score_doc(d, question), reverse=True)
    return scored[:top_n]

def _highlight_keywords(text: str, question: str, max_chars: int = CITATION_HIGHLIGHT_CHARS) -> str:
    q_tokens = [t for t in re.findall(r'\w{3,}', question.lower())]
    best_pos = 0
    best_score = -1
    window = max_chars
    step = max(50, window // 10)

    for i in range(0, max(1, len(text) - window + 1), step):
        chunk = text[i:i + window].lower()
        score = sum(1 for t in q_tokens if t in chunk)
        if score > best_score:
            best_score = score
            best_pos = i

    snippet = text[best_pos:best_pos + max_chars]
    prefix = '...' if best_pos > 0 else ''
    suffix = '...' if best_pos + max_chars < len(text) else ''

    snippet = ( snippet
               .replace('&', '&amp;')
               .replace('<', '&lt;')
               .replace('>', '&gt;'))
    
    for token in sorted(q_tokens, key=len, reverse=True):
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        snippet = pattern.sub(
            lambda m: (
                f'<mark style="background:#1a4a2a;color:#00ff88;'
                f'border-radius:2px;padding:0 2px;">{m.group()}</mark>'
            ),
            snippet,
        )

    return prefix + snippet + suffix

def render_citations(source_docs: list, question: str = ''):
    if not source_docs:
        return
    
    if not question:
        question = (
            st.session_state.get('multi_last_question')
            or st.session_state.get('last_question')
            or ''
        )

    top_docs = _select_top_docs(source_docs, question, top_n=CITATION_TOP_N)
    total_raw = len(_dedup_docs(source_docs))
    
    st.markdown('---')
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#0099ff;'
        f'text-transform:uppercase;letter-spacing:0.08em;">'
        f'Sources <span style="color:#4a6a8a;font-weight:400;">'
        f'(top {len(top_docs)} / {total_raw} retrieved)</span></p>',
        unsafe_allow_html=True,
    )

    for i, doc in enumerate(top_docs, 1):
        page = _get_page(doc)
        source_file = _get_source_file(doc)
        preview = _make_preview(doc.page_content)
        score = _score_doc(doc, question)
        score_pct = int(score * 100)

        if score_pct >= 60:
            bar_color = '#00cc66'
        elif score_pct >= 30:
            bar_color = '#f0a000'
        else:
            bar_color = '#4a6a8a'

        score_html = (
            f'<span style="font-size:0.65rem;color:{bar_color};margin-left:0.5rem;">'
            f'{score_pct}% match</span>'
        )

        if source_file:
            label = f'[{i}] {source_file} · Page {page} — {preview}'
            header_html = (
                f'<div class="sd-citation-header">'
                f'Citation [{i}] &nbsp;·&nbsp; '
                f'<span style="color:#00cc66;">{source_file}</span>'
                f' &nbsp;·&nbsp; Page {page}'
                f'{score_html}'
                f'</div>'
            )
        else:
            label = f'[{i}] Page {page} — {preview}'
            header_html = (
                f'<div class="sd-citation-header">'
                f'Citation [{i}] · Page {page}'
                f'{score_html}'
                f'</div>'
            )

        highlighted_body = _highlight_keywords(doc.page_content, question)

        with st.expander(label, expanded=False):
            st.markdown(header_html, unsafe_allow_html=True)
            st.markdown(
                f'<div class="sd-citation-body">{highlighted_body}</div>',
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
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
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
    max-height: 280px;
    overflow-y: auto;
}
</style>
'''