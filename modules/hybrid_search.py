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


def render_hybrid_toggle():
    st.subheader('Search Mode ')

    use_hybird = st.toggle(
        'Hybrid search (BM25 + Vector)',
        value=st.session_state.get('use_hybrid', False),
        key='toggle_hybrid',
        help=(
            'ON -> combiness keyword search (BM25) with semantic search (FAISS)\n'
            'OFF -> semantic search only (default)'
        ),
    )

    st.session_state['use_hybrid'] = use_hybird

    if use_hybird:
        st.markdown(
            '<span style="font-size:0.75rem;color:#7aaed0;">'
            "BM25 40% · Vector 60%</span>",
            unsafe_allow_html=True,
        )

    return use_hybird