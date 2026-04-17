from email.policy import default

import streamlit as st
from config import CHUNK_SIZE, CHUNK_OVERLAP

def init_session():
    default = {
        'vector_store': None,
        'retriever': None,
        'doc_name': None,
        'doc_chunks': 0,
        'doc_bytes': None,
        'raw_docs': None,
        'avg_chunk_len': 0,
        'upload_key': 0,

        'chat_history': [],
        
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'chunk_metrics': [],
        
        'conv_memory': [],

        'use_hybrid': False,
        'hybrid_retriever': None,

        'multi_vector_store': None,
        'multi_retriever': None,
        'doc_registry': {},

        'use_rerank': False,

        'use_self_rag': False,
        
    }

    for key, value in default.items():
        if key not in st.session_state:
            st.session_state[key] = value