import time
from config import RETRIEVER_K
import streamlit as st 

_cross_encoder = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 7

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

def retrieve_and_rerank(question: str, retriever, fetch_k: int = 20) -> list:
    original_k = None
    search_kwargs = getattr(retriever, 'search_kwargs', None)
    if search_kwargs is not None:
        original_k = search_kwargs.get('k', RETRIEVER_K)
        search_kwargs['k'] = fetch_k

    t0 = time.time()
    candidates = retriever.invoke(question)
    retrieval_ms = (time.time() - t0) * 1000

    if search_kwargs is not None and original_k is not None:
        search_kwargs['k'] = original_k

    t1 = time.time()
    reranked = rerank(question, candidates)
    rerank_ms = (time.time() - t1) * 1000

    st.session_state['rerank_metrics'] = {
        'retrieval_ms': round(retrieval_ms),
        'rerank_ms': round(rerank_ms),
        'candidates': len(candidates),
        'top_n': len(reranked),
    }

    return reranked

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
        metrics = st.session_state.get('rerank_metrics')
        if metrics:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#4a6a8a;">'
                f"Last: retrieve {metrics['retrieval_ms']}ms · "
                f"rerank {metrics['rerank_ms']}ms · "
                f"{metrics['candidates']} → {metrics['top_n']} docs"
                f"</span>",
                unsafe_allow_html=True,
            )

    return use_rerank