import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile
from gtts import gTTS
import io
import time
from huggingface_hub import login

# Login to Hugging Face (removes warning)
from huggingface_hub import login
login("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")  # ← replace with your token or use os.getenv("HF_TOKEN")

st.set_page_config(page_title="AI Lecturer", layout="wide")

st.title("🎓 AI Lecturer")
st.caption("Upload textbooks or use pre-loaded ones → Ask questions → Get explanations, references & voice")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = "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"  
    level = st.selectbox("Explanation Level", ["Beginner", "Intermediate", "Advanced"], index=1)
    course = st.text_input("Course", "Biology / Medicine")
    topic = st.text_input("Focus Topic", "Lung Cancer")

if not api_key:
    st.info("Enter Groq API key.")
    st.stop()

uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.sidebar.button("Process / Load Textbooks"):
    docs = []
    sources = []

    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())
            sources.append(f"Uploaded: {file.name}")
    else:
        textbook_dir = "textbooks"
        if os.path.exists(textbook_dir):
            for filename in os.listdir(textbook_dir):
                if filename.lower().endswith(".pdf"):
                    path = os.path.join(textbook_dir, filename)
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                    sources.append(f"Pre-loaded: {filename}")

    if not docs:
        st.error("No PDFs available.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="lecturer_docs"
        )
        st.sidebar.success(f"Loaded {len(chunks)} chunks")
        st.sidebar.write("Sources:")
        for s in sources:
            st.sidebar.write(f"- {s}")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = "No textbook loaded."
            references = []

            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                for doc in relevant_docs:
                    source = doc.metadata.get("source", "Unknown")
                    snippet = doc.page_content[:150] + "..."
                    references.append(f"{source}: {snippet}")

            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)

            system_prompt = f"""You are a lecturer teaching {course} on {topic}.
Explain at {level} level.
Use textbook content when available.

Context: {context}"""

            messages = [HumanMessage(content=system_prompt)] + [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[-8:]
            ]

            response = llm.invoke(messages)
            st.markdown(response.content)

            if references:
                with st.expander("📚 References"):
                    for ref in references:
                        st.write(ref)

            st.session_state.last_response = response.content
            st.session_state.messages.append({"role": "assistant", "content": response.content})

    if st.button("🎤 Voice Reply"):
        st.info("Voice input coming soon — type for now.")

# Voice output
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""

if st.button("🔊 Listen to Response"):
    if st.session_state.last_response.strip():
        with st.spinner("Generating voice..."):
            tts = gTTS(st.session_state.last_response, lang='en', slow=True)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf, format="audio/mp3")
    else:
        st.warning("No response yet.")

# Video
if st.button("🎥 Find video"):
    if st.session_state.messages:
        q = st.session_state.messages[-1]["content"]
        query = q.replace(" ", "+") + "+explained+simply"
        url = f"https://www.youtube.com/results?search_query={query}"
        st.markdown(f"[YouTube videos]({url})")
    else:
        st.info("Ask first.")

st.caption("Built in Kampala • Groq + LangChain")

