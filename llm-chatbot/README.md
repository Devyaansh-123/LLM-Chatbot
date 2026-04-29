# 🤖 LLM Chatbot — RAG-Powered PDF Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot built with Streamlit that lets you upload PDF documents and ask natural language questions about them. Powered by LLaMA-3.3 (via Groq) or GPT-4o-mini (via OpenAI), with semantic search, text-to-speech responses, and a smooth typing animation.

---

## ✨ Features

- 📄 **Multi-PDF Upload** — Upload one or more PDF files and query across all of them
- 🔍 **Semantic Search** — Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings + ChromaDB for fast retrieval
- 🦙 **Dual LLM Support** — Auto-selects **Groq LLaMA-3.3-70b** (preferred) or **OpenAI GPT-4o-mini** based on your API keys
- 🔊 **Text-to-Speech** — Hear answers read aloud via Google TTS
- ⌨️ **Typing Animation** — Responses stream character-by-character with sound effects
- 📜 **Chat History** — Full session history in the sidebar with expandable Q&A
- 🎬 **Lottie Animations** — Smooth loading animations while the model thinks

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit |
| **LLM** | Groq (LLaMA-3.3-70b) / OpenAI (GPT-4o-mini) |
| **RAG Framework** | LangChain |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | ChromaDB |
| **PDF Parsing** | PyMuPDF |
| **TTS** | gTTS (Google Text-to-Speech) |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Devyaansh-123/LLM-Chatbot.git
cd LLM-Chatbot/llm-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the `llm-chatbot/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional fallback
```

> 💡 Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 How It Works

```
PDF Upload → Text Extraction (PyMuPDF)
         → Chunking & Embedding (MiniLM-L6-v2)
         → Vector Storage (ChromaDB)
         → Query → Semantic Retrieval (top-5 chunks)
         → LLM Answer Generation (LLaMA / GPT)
         → Typed Response + TTS Audio
```

1. You upload a PDF — it's parsed and split into text chunks
2. Each chunk is embedded into a vector using `sentence-transformers`
3. When you ask a question, the top-5 most relevant chunks are retrieved
4. The LLM uses those chunks as context to generate a grounded answer
5. The answer is streamed with a typing effect and read aloud

---

## 📁 Project Structure

```
llm-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── data/               # Sample PDF documents
│   ├── Adrian-Rosebrock-Deep-Learning-for.pdf
│   └── oreilly_chapter_excerpt_nlpt.pdf
├── model_cache/        # Auto-downloaded embedding model (not committed)
└── db/                 # ChromaDB vector store (not committed)
```

---

## ⚙️ Configuration

| Env Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (uses LLaMA-3.3-70b) — **preferred** |
| `OPENAI_API_KEY` | OpenAI API key (uses GPT-4o-mini) — fallback |

The app auto-detects which key is available and picks the best LLM accordingly.

---

## 📦 Requirements

- Python 3.9+
- Internet connection (for first-time model download and LLM API calls)

---

## 📄 License

MIT License — feel free to use and modify.

---

*Built with ❤️ using LangChain, Streamlit, and Groq*
