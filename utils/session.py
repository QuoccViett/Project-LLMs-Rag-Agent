from email.policy import default

import streamlit as st
from config import CHUNK_SIZE, CHUNK_OVERLAP

def init_session():
    default = {
        'vector_store': None,
        'retriever': None,
        'doc_name': None,
        'doc_chunks': 0,
        'chat_history': [],
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'chunk_metrics': [],
        'conv_memory': [],
        'upload_key': 0,
    }

    for key, value in default.items():
        if key not in st.session_state:
            st.session_state[key] = value