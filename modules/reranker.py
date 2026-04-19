from calendar import c

import streamlit as st 

_cross_encoder = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 3

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

def rerank(question: str, candidate_docs: list, top_n: int = RERANK_TOP_N) -> list:
    if not candidate_docs:
        return candidate_docs
    
    cross_encoder = _get_cross_encoder()

    pairs = [(question, doc.page_content) for doc in candidate_docs]

    scores = cross_encoder.predict(pairs)

    scored = sorted(
        zip(scores, candidate_docs),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in scored[:top_n]]

def retrieve_and_rerank(question: str, retriever, fetch_k: int = 10) -> list:
    original_k = retriever.search_kwargs.get('k', 3)
    retriever.search_kwargs['k'] = fetch_k

    candidates = retriever.invoke(question)

    retriever.search_kwargs['k'] = original_k

    return rerank(question, candidates)

def render_rerank_toggle():
    st.subheader('Re-ranking')
    use_rerank = st.toggle(
        'Cross-Encoder Re_Ranking',
        value=st.session_state.get('use_rerank', False),
        key='toggle_rerank',
        help=(
            'ON -> fetches 10 candidates, cross-encoder re-scores and picks top 3\n'
            'OFF -> standard FAISS top-3 (faster)'
        ),
    )
    st.session_state['use_rerank'] = use_rerank

    if use_rerank:
        st.markdown(
            '<span style="font-size:0.75rem;color:#7aaed0;">'
            f"Model: {CROSS_ENCODER_MODEL}</span>",
            unsafe_allow_html=True,
        )

    return use_rerank