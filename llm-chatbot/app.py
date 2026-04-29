import os
import time
import tempfile
import streamlit as st
import requests
import io
from dotenv import load_dotenv
from gtts import gTTS
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader  

 # LLM
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


# helper
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Failed to load Lottie: {e}")
    return None


# setup
load_dotenv()
st.set_page_config(page_title="LLM Chatbot", layout="wide")
st.title(" LLM Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "files_ready" not in st.session_state:
    st.session_state.files_ready = False

# ---------- Sidebar ----------
st.sidebar.title(" Chat History")
if st.sidebar.button(" Start Over"):
    st.session_state.chat_history = []
    st.session_state.files_ready = False

if st.session_state.chat_history:
    for idx, item in enumerate(reversed(st.session_state.chat_history)):
        with st.sidebar.expander(f"Q{len(st.session_state.chat_history)-idx}: {item['question'][:50]}..."):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
else:
    st.sidebar.info("No chat history yet.")


# ---------- Upload PDFs ----------
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and not st.session_state.files_ready:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Click below when ready.")
    if st.button(" Done Uploading"):
        st.session_state.files_ready = True


# ---------- Process PDFs ----------
@st.cache_resource
def get_vectorstore(_docs, _embedding):
    return FAISS.from_documents(_docs, _embedding)

def run_qa(query, retriever, llm):
    """Direct retrieval + LLM call — no langchain.chains dependency."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Answer the question based only on "
            "the following context. If the answer is not in the context, say so.\n\n"
            + context
        )),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return {"result": response.content, "source_documents": docs}

if st.session_state.files_ready and uploaded_files:
    docs = []
    with st.status("📄 Processing PDFs...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyMuPDFLoader(tmp_path)
            pdf_docs = loader.load()

            if not pdf_docs or all(len(d.page_content.strip()) == 0 for d in pdf_docs):
                st.error(f"⚠️ {uploaded_file.name} has no readable text (may be a scanned image).")
                st.stop()

            docs.extend(pdf_docs)
        
        st.write(f" Extracted {len(docs)} text chunks.")

        # ---------- Embeddings ----------
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./model_cache"
        )

        db = get_vectorstore(docs, embedding)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        status.update(label=" Database Ready!", state="complete", expanded=False)

    # ---------- Auto-select LLM ----------
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key and ChatGroq is not None:
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
            )
            st.sidebar.success(" Using Groq LLaMA-3.3")
        except Exception as e:
            st.sidebar.warning(f"Groq unavailable ({e}). Switching to OpenAI.")
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key)
    elif openai_key and ChatOpenAI is not None:
        st.sidebar.success("⚡ Using OpenAI GPT-4o-mini")
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key)
    else:
        st.error("❌ No valid API key found. Add GROQ_API_KEY or OPENAI_API_KEY in your .env file.")
        st.stop()

    # ---------- Retrieval QA ----------

    st.divider()
    query = st.text_input("💬 Ask a question about your documents:", placeholder="e.g., What are the main findings?")

    if query:
        with st.spinner("🤖 Thinking..."):
            lottie_placeholder = st.empty()
            anim = load_lottieurl("https://lottie.host/9c83c08d-58d2-4ff6-87d3-1d6dcf2b6d1e/Hk94Hb9rm5.json")
            if anim:
                with lottie_placeholder:
                    st_lottie(anim, height=250)

            result = run_qa(query, retriever, llm)
            full_response = result["result"]
            lottie_placeholder.empty()

            st.session_state.chat_history.append({
                "question": query,
                "answer": full_response
            })

            response_placeholder = st.empty()
            typed = ""

            components.html(
                """
                <audio id="typeSound" src="https://cdn.pixabay.com/audio/2023/03/28/audio_f96b958309.mp3" preload="auto"></audio>
                <script>
                function playTypingSound() {
                    const a = document.getElementById("typeSound");
                    if (a) { a.volume = 0.4; a.currentTime = 0; a.play(); }
                }
                </script>
                """,
                height=0,
            )

            for i, char in enumerate(full_response):
                typed += char
                if i % 3 == 0:
                    components.html("<script>playTypingSound();</script>", height=0)
                response_placeholder.markdown(f"🤖 {typed}▌" if i % 2 == 0 else f"🤖 {typed}")
                time.sleep(0.01)

            response_placeholder.markdown(f"🤖 {typed}")

            # ---------- Text-to-Speech ----------
            tts = gTTS(typed, lang="en")
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            st.markdown("🔊 **Click to hear the answer:**")
            st.audio(audio_io.read(), format="audio/mp3")

            # ---------- Retrieved context ----------
            with st.expander("📄 Retrieved context from your PDF"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Chunk {i}:**\n\n{doc.page_content[:800]}...\n---")















 