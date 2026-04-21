import time
import streamlit as st 
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config import RETRIEVER_K


def build_hybrid_retriever(documents: list, vector_store):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVER_K

    vector_retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': RETRIEVER_K},
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    return ensemble


def get_hybrid_retriever_or_fallback():
    hybrid = st.session_state.get('hybrid_retriever')
    if hybrid is not None:
        return hybrid
    return st.session_state.retriever


def build_and_store_hybrid(documents: list, vector_store):
    hybrid = build_hybrid_retriever(documents, vector_store)
    st.session_state['hybrid_retriever'] = hybrid
    return hybrid

def compare_hybrid_vs_vector(question: str, k: int = RETRIEVER_K) -> dict | None:
    hybrid = st.session_state.get('hybrid_retriever')
    vector = st.session_state.get('retriever')

    if hybrid is None or vector is None:
        return None
    
    t0 = time.time()
    hybrid_docs = hybrid.invoke(question)
    hybrid_ms = round((time.time() -t0) * 1000)

    t1 = time.time()
    vector_docs = vector.invoke(question)
    vector_ms = round((time.time() - t1) *1000)

    hybrid_ids = {id(d) for d in hybrid_docs}
    vector_ids = {id(d) for d in vector_docs}
    unique_to_hybrid = len(hybrid_ids - vector_ids)

    return {
        'hybrid_ms': hybrid_ms,
        'vector_ms': vector_ms,
        'hybrid_docs': len(hybrid_docs),
        'vector_docs': len(vector_docs),
        'unique_to_hybrid': unique_to_hybrid,
    }


def render_hybrid_toggle(key_suffix=''):
    st.subheader('Search Mode')

    use_hybird = st.toggle(
        'Hybrid search (BM25 + Vector)',
        value=st.session_state.get('use_hybrid', False),
        key=f'toggle_hybrid{key_suffix}',
        help=(
            'ON -> combiness keyword search (BM25) with semantic search (FAISS)\n'
            'OFF -> semantic search only (default)'
        ),
    )

    st.session_state['use_hybrid'] = use_hybird

    if use_hybird:
        st.markdown(
            '<span style="font-size:0.75rem;color:#7aaed0;">'
            "BM25 40% · Vector 60%"
            "</span>",
            unsafe_allow_html=True,
        )
        cmp = st.session_state.get('hybrid_compare')
        if cmp:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#4a6a8a;">'
                f"Last: hybrid {cmp['hybrid_ms']}ms vs vector {cmp['vector_ms']}ms · "
                f"+{cmp['unique_to_hybrid']} unique docs from BM25"
                f"</span>",
                unsafe_allow_html=True,
            )

    return use_hybird