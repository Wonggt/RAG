# 🧠 RAG-Based AI Assistant with Streamlit and TTS

A lightweight, local-ready **Retrieval-Augmented Generation (RAG)** application that combines:
- 📚 LangChain for LLM orchestration
- 🎤 Microsoft Edge TTS for text-to-speech responses
- ⚡ FAISS for fast document retrieval
- 🖥️ Streamlit for interactive UI

---

## 🚀 Features

- Upload your own documents and ask questions with context-aware responses.
- Voice responses using Microsoft Edge TTS.
- Streamlit-based web UI for local or online deployment.
- Supports OpenAI, Azure, and Claude models via LangChain.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Wonggt/RAG.git
cd RAG-main
```

### 2. Create and activate virtual environment

Using conda:
```bash
conda create -n rag_app python=3.9
conda activate rag_app
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you encounter FAISS install issues, try:
> 
> ```bash
> pip install faiss-cpu
> ```

---

## 🧪 Running the App

```bash
streamlit run app.py
```

This will open a local Streamlit app in your browser at `http://localhost:8501`.

---

## 🧩 Project Structure

```
rag_llm_app/
│
├── app.py              # Main Streamlit UI
├── rag_methods.py      # Core RAG logic (retrieval, LLM)
├── tts.py              # Text-to-speech handler using edge-tts
├── requirements.txt    # Project dependencies
└── README.md
```

---

## 🔐 API Keys

To use OpenRouter, set your API keys via environment variables or `.env` file:

```
OPENROUTER_API_KEY=your-openrouter-key
```

---

## 📄 License

MIT License

---

## 🙋‍♀️ Contributing

Pull requests and suggestions welcome! Please open an issue first to discuss what you'd like to change.

---

## ⭐ Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Edge TTS](https://github.com/rany2/edge-tts)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://github.com/streamlit/streamlit)
