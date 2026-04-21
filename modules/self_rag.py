import re
import streamlit as st
from config import ( 
    RETRIEVER_K, EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP        
)

CONFIDENCE_THRESHOLD = 60
MAX_RETRIES = 1

_SYSTEM_INFO = f"""
=== THÔNG TIN KIẾN TRÚC HỆ THỐNG ADI ===
- Embedding model  : {EMBEDDING_MODEL}
- Số chiều vector  : 768  (paraphrase-multilingual-mpnet-base-v2)
- LLM model        : {LLM_MODEL}
- Vector database  : FAISS (cosine similarity)
- Framework        : LangChain + Streamlit
- Chunk size mặc định  : {CHUNK_SIZE} ký tự
- Chunk overlap mặc định: {CHUNK_OVERLAP} ký tự
- Số chunks retrieve (k): {RETRIEVER_K}
- Hybrid search    : BM25 (40%) + Vector FAISS (60%) qua EnsembleRetriever
- Re-ranking       : CrossEncoder ms-marco-MiniLM-L-6-v2
""".strip()

_SYSTEM_KW = [
    "embedding", "vector", "chiều", "dimension", "mpnet",
    "faiss", "chunk", "overlap", "retriever", "llm", "model", "qwen",
    "kiến trúc", "hệ thống", "architecture", "framework", "langchain",
    "paraphrase", "sentence-transformer", "768", "cosine",
    "bm25", "hybrid", "cross-encoder", "rerank",
]

def _is_system_query(question: str) -> bool:
    q=question.lower()
    return any(kw in q for kw in _SYSTEM_KW)

def _llm_text(response) -> str:
    if hasattr(response, 'context'):
        return response.content.strip()
    return str(response).strip()

def _rewrite_query(question: str, llm) -> str:
    prompt = (
        "Rewrite the following question to be clearer and more specific "
        "for a document search. Output ONLY the rewritten question, "
        "no explanation, no quotes.\n\n"
        f"Original: {question}\n\n"
        "Rewritten:"
    )
    try:
        rewritten = _llm_text(llm.invoke(prompt))
        if not rewritten or len(rewritten) < 5 or '\n' in rewritten or len(rewritten) > 300:
            return question
        return rewritten
    except Exception:
        return question
    
def _generate_answer(context: str, question: str, llm) -> str:
    prompt = (
        "Use the following document context to answer the question. "
        "If the answer is not in the context, say 'I could not find this in the document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return llm.invoke(prompt)

def _parse_evaluation(raw: str) -> dict:
    result = {'confidence': 50, 'relevance': 'medium', 'groundedness': 'partial'}

    conf_match = re.search(r'CONFIDENCE[:\s]+(\d+)', raw, re.IGNORECASE)
    if conf_match:
        val = int(conf_match.group(1))
        result['confidence'] = max(0, min(100, val))

    rel_match = re.search(r'RELEVANCE[:\s]+(high|medium|low)', raw, re.IGNORECASE)
    if rel_match:
        result['relevance'] = rel_match.group(1).lower()

    gnd_match = re.search(
        r'GROUNDEDNESS[:\s]+(grounded|partial|not grounded)', raw, re.IGNORECASE
    )
    if gnd_match:
        result['groundedness'] = gnd_match.group(1).lower()

    return result
        

def _evaluate(question: str, context: str, answer: str, llm) -> dict:
    eval_prompt = (
        "Evaluate the following Q&A pair. "
        "Reply with EXACTLY these 3 lines, no extra text:\n"
        "CONFIDENCE: <number 0-100>\n"
        "RELEVANCE: <high|medium|low>\n"
        "GROUNDEDNESS: <grounded|partial|not grounded>\n\n"
        f"Question: {question}\n"
        f"Context (excerpt): {context[:800]}\n"
        f"Answer: {answer}\n\n"
        "Evaluation:"
    )
    try:
        raw = _llm_text(llm.invoke(eval_prompt))
        return _parse_evaluation(raw)
    except Exception:
        return {'confidence': 50, 'relevance': 'medium', 'groundedness': 'partial'}

def self_rag_answer(question: str, retriever, llm) -> dict:
    rewritten = _rewrite_query(question, llm)

    if _is_system_query(question):
        context = _SYSTEM_INFO
        source_docs = []
    else:
        source_docs = retriever.invoke(rewritten)
        context = '\n\n'.join(doc.page_content for doc in source_docs)

    answer = _generate_answer(context, question, llm)
    evaluation = _evaluate(question, context, answer, llm)

    retried = False

    if evaluation['confidence'] < CONFIDENCE_THRESHOLD and not _is_system_query(question):
        retried = True
        source_docs = retriever.invoke(question)
        context = '\n\n'.join(doc.page_content for doc in source_docs)
        answer = _generate_answer(context, question, llm)
        evaluation = _evaluate(question, context, answer, llm)

    return {
        "answer":       answer,
        "source_docs":  source_docs,
        "rewritten_q":  rewritten,
        "confidence":   evaluation["confidence"],
        "relevance":    evaluation["relevance"],
        "groundedness": evaluation["groundedness"],
        "retried":      retried,
    }

def render_self_rag_toggle():
    st.subheader('Self-RAG')
    use_self_rag = st.toggle(
        'Self-RAG (Query Rewrite + Confidence)', 
        value=st.session_state.get('use_self_rag', False),
        key='toggle_self_rag',
        help=(
            "ON -> rewrites your question, evaluates answer quality,"
            "show confidence score\n"
            "OFF -> standard RAG pipeline"
        ),
    )
    st.session_state['use_self_rag'] = use_self_rag
    return use_self_rag

def render_self_rag_metadata(result: dict):
    confidence = result.get('confidence', 0)
    relevance = result.get('relevance', '-')
    groundedness = result.get('groundedness', '-')
    rewritten_q = result.get('rewritten_q', '')
    retried = result.get('retried', False)

    if confidence >= 75:
        conf_color = "#00cc66"
    elif confidence >= 50:
        conf_color = "#f0a000"
    else:
        conf_color = "#cc3333"
 
    st.markdown(
        f"""
        <div style="
            background:#0a151f;
            border:1px solid #1e3a5f;
            border-radius:8px;
            padding:0.8rem 1rem;
            margin-top:0.6rem;
            font-size:0.82rem;
            color:#7aaed0;
        ">
            <span style="color:#0099ff;font-family:'IBM Plex Mono',monospace;
                         text-transform:uppercase;font-size:0.7rem;">
                Self-RAG evaluation
            </span><br><br>
            <b>Confidence:</b>
            <span style="color:{conf_color};font-weight:700;">{confidence}/100</span>
            &nbsp;·&nbsp;
            <b>Relevance:</b> {relevance}
            &nbsp;·&nbsp;
            <b>Grounded:</b> {groundedness}
            {"&nbsp;·&nbsp;<span style='color:#f0a000;'>retried with original query</span>" if retried else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
