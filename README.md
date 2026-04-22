# Project LLMs RAG AGENT - RAG System
Hệ thống hỏi đáp tài liệu thông minh sử dụng RAG (Retrieval-Augmented Generation).

## Công nghệ sử dụng
- **LLM:** Qwen2.5:7b (qua Ollama)
- **Framework:** Streamlit, LangChain
- **Vector DB:** FAISS
- **Embedding:** MPNet (paraphrase-multilingual-mpnet-base-v2)

## Cách cài đặt
Tạo môi trường ảo: `python -m venv venv`
Kích hoạt venv: `/venv/Scripts/activate`
Cài thư viện: `pip install -r requirements.txt`
Cài đặt Ollama và pull model: `ollama pull qwen2.5:7b`
Chạy ứng dụng: `python -m streamlit run app.py`