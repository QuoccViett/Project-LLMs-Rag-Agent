import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P,
    LLM_REPEAT_PENALTY
)


def _resolve_embedding_device(requested: str) -> str:
    req = (requested or '').strip().lower()
    is_auto = req in ('', 'auto')

    if is_auto:
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    if req.startswith('cuda'):
        try:
            import torch
            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        st.warning(
            "CUDA embedding requested but not available. Falling back to CPU. "
            "(Tip: set EMBEDDING_DEVICE='cpu' in config.py if you don't have CUDA.)"
        )
        return 'cpu'

    return requested

@st.cache_resource(show_spinner=False)
def load_embedder() -> HuggingFaceEmbeddings:
    device = _resolve_embedding_device(EMBEDDING_DEVICE)
    with st.spinner(f"Loading embedding model ({device}) ..."):
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
        )

@st.cache_resource(show_spinner='Connecting to Ollana LLM...')
def load_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        top_p = LLM_TOP_P,
        repeat_penalty=LLM_REPEAT_PENALTY,
        num_gpu=99,
    )