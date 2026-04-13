import time
import streamlit as st

def init_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_to_history(question: str, answer: str):
    st.session_state.chat_history.append({
        'question': question,
        'answer': answer,
        'timestamp': time.strftime("%H:%M %d %b %Y"),
    })

def render_history_sidebar():
    st.subheader('Chat History')

    history = st.session_state.get('chat_history', [])

    if not history:
        st.caption('No question yet.')
        return
    
    for i, item in enumerate(reversed(history)):
        idx = len(history) - i
        label = f"#{idx} {item['question'][:38]}{'...' if len(item['question']) > 38 else ''}"
        with st.expander(label, expanded=False):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.caption(f" {item['timestamp']}")