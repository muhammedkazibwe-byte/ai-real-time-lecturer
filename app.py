import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from gtts import gTTS
import io
import os
import tempfile

# ───────────────────────────────────────────────
# Global variables
vectorstore = None

def load_textbooks(uploaded_files=None):
    global vectorstore
    docs = []

    # Priority: uploaded files
    if uploaded_files:
        for file_path in uploaded_files:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
    # Fallback: pre-loaded folder
    else:
        textbook_dir = "textbooks"
        if os.path.exists(textbook_dir):
            for filename in os.listdir(textbook_dir):
                if filename.lower().endswith(".pdf"):
                    path = os.path.join(textbook_dir, filename)
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())

    if not docs:
        return "No PDFs found. Upload files or add PDFs to 'textbooks/' folder."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="lecturer_docs"
    )

    return f"Loaded {len(chunks)} chunks successfully."

def chat(message, history, api_key, explanation_level, course, topic):
    if not api_key:
        yield "Please enter your Groq API key.", None

    if not vectorstore:
        yield "Please process textbooks first.", None

    # Build context from RAG
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(message)
    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7
    )

    system_prompt = f"""You are a patient, clear university lecturer teaching {course} with focus on {topic}.
Explain at {explanation_level} level.
Always base answers on the provided textbook content when available.
Be encouraging and ask follow-up questions to check understanding.

Textbook context:
{context}"""

    messages = [HumanMessage(content=system_prompt)] + [
        HumanMessage(content=m[0]) if m[0] else AIMessage(content=m[1])
        for m in history
    ] + [HumanMessage(content=message)]

    # Stream response
    full_response = ""
    for chunk in llm.stream(messages):
        full_response += chunk.content
        yield full_response, None

    # Save last response for voice
    global last_response
    last_response = full_response

    yield full_response, None

def speak_response():
    global last_response
    if not last_response or not last_response.strip():
        return None

    sentences = [s.strip() for s in last_response.split('. ') if s.strip()]
    audio_files = []

    for sentence in sentences:
        clean = sentence + "."
        tts = gTTS(clean, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_files.append(buf)

    # Return first sentence for now (Gradio can only play one at a time)
    # You can extend to concatenate or play sequentially in frontend
    return audio_files[0] if audio_files else None

def video_search(message):
    if not message:
        return "Ask a question first."

    query = message.strip()[:120].replace(" ", "+") + "+explained+simply"
    url = f"https://www.youtube.com/results?search_query={query}"
    return f"**[Watch explanatory videos on YouTube]({url})**"

# ───────────────────────────────────────────────
# Gradio Interface
with gr.Blocks(title="AI Real-Time Lecturer") as demo:
    gr.Markdown("# 🎓 AI Real-Time Lecturer")
    gr.Markdown("Upload textbooks or use the `textbooks/` folder → Ask questions → Get voice answers")

    with gr.Row():
        api_key = gr.Textbox(label="Groq API Key", type="password")
        explanation_level = gr.Dropdown(
            choices=["Beginner", "Intermediate", "Advanced"],
            value="Intermediate",
            label="Explanation Level"
        )
    course = gr.Textbox(label="Course", value="Biology / Medicine")
    topic = gr.Textbox(label="Topic", value="Lung Cancer")

    pdf_upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs (optional)")
    load_btn = gr.Button("Process Textbooks / Load Pre-collected")

    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Type your question")
    voice_btn = gr.Button("🎤 Speak Question (5 sec record)")
    voice_audio = gr.Audio(source="microphone", type="filepath", label="Speak now")

    with gr.Row():
        send_btn = gr.Button("Send")
        speak_btn = gr.Button("🔊 Listen to Response")
        video_btn = gr.Button("🎥 Find Video")

    # ─── Event Handlers ──────────────────────────────────────────

    load_btn.click(
        fn=load_textbooks,
        inputs=[pdf_upload],
        outputs=status
    )

    def user_speak(audio_path):
        if not audio_path:
            return "", ""
        # Whisper transcription would go here
        # For now placeholder
        return "Transcribed: (implement Whisper here)", ""

    voice_btn.click(
        fn=user_speak,
        inputs=voice_audio,
        outputs=[msg, status]
    )

    def submit_chat(message, history, api_key, level, course, topic):
        if not message.strip():
            return history, ""
        # Simulate response (replace with real chat fn)
        history.append((message, "Thinking..."))
        return history, ""

    send_btn.click(
        fn=chat,
        inputs=[msg, chatbot, api_key, explanation_level, course, topic],
        outputs=[chatbot, status]
    )

    speak_btn.click(
        fn=speak_response,
        outputs=gr.Audio(label="AI Voice Response")
    )

    video_btn.click(
        fn=video_search,
        inputs=msg,
        outputs=status
    )

demo.launch()
