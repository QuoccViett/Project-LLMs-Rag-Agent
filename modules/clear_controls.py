import streamlit as st

def _confirm_action(key: str, label: str, on_confirm):
    if st.button(label, use_container_width=True, key=f'btn_{key}'):
        st.session_state[f'pending_{key}'] = True
        
    if st.session_state.get(f'pending_{key}'):
        st.warning('Are you sure? This cannot be undone.')
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Confirm', key=f'yes_{key}'):
                on_confirm()
                st.session_state[f'pending_{key}'] = False
                st.rerun()
        with col2:
            if st.button('Cancel', key=f'no_{key}'):
                st.session_state[f'pending_{key}'] = False
                st.rerun()

def _clear_history():
    st.session_state.chat_history = []
    st.session_state.conv_memory = []
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.session_state.last_sources = []
    st.toast('Chat history cleared.')

def _clear_document():
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.doc_name = None
    st.session_state.doc_chunks = 0
    st.session_state.chat_history = []
    st.session_state.conv_memory = []
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.session_state.last_sources = []
    st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
    st.toast('Document and history cleared.')

def render_clear_controls():
    st.subheader('Clear data')
    _confirm_action('history', 'Clear Chat History', _clear_history)
    _confirm_action('document', 'Clear Uploaded Document', _clear_document)