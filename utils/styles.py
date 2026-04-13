import streamlit as st

_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}


.sd-header {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.sd-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0066cc, #00ccff, #0066cc);
}
.sd-header h1 {
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    color: #e8f4ff;
    letter-spacing: -0.02em;
}
.sd-header p {
    margin: 0.25rem 0 0;
    color: #7aa3c8;
    font-size: 0.85rem;
}
.sd-badge {
    display: inline-block;
    background: #1a2e42;
    border: 1px solid #2a4a6a;
    color: #7cc8ff;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    margin: 6px 4px 0 0;
}


.sd-answer {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #0099ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #d0e8ff;
    font-size: 0.95rem;
    line-height: 1.65;
    margin-top: 0.8rem;
}
.sd-answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #0099ff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}


.sd-stats {
    display: flex;
    gap: 8px;
    margin-top: 6px;
}
.sd-stat {
    flex: 1;
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.5rem;
    text-align: center;
}
.sd-stat strong {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    color: #0099ff;
}
.sd-stat span {
    font-size: 0.72rem;
    color: #7aa3c8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}


.sd-source {
    background: #090f16;
    border: 1px solid #1a2e42;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #5a8ab0;
    margin-bottom: 0.5rem;
    white-space: pre-wrap;
    word-break: break-word;
}

 
footer { visibility: hidden; }
</style>
'''

def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)