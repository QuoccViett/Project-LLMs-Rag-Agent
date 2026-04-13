import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P,
    LLM_REPEAT_PENALTY
)

@st.cache_resource(show_spinner='Loading embedding model (GPU) ...')
def load_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {'device': EMBEDDING_DEVICE},
        encode_kwargs = {'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner='Connecting to Ollana LLM...')
def load_llm() -> OllamaLLM:
    return OllamaLLM(
        model=LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        top_p = LLM_TOP_P,
        repeat_penalty=LLM_REPEAT_PENALTY
    )