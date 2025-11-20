# app.py

import streamlit as st
from rag_pipeline import load_documents, create_temp_qa_chain, generate_suggestions, extract_legal_glossary
import tempfile, os, io
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
import streamlit.components.v1 as components
from html import escape
from transformers import pipeline

# ------------------- Page Config -------------------
st.set_page_config(page_title="Justify", layout="centered")
st.markdown("""
    <style>
    .centered-title { text-align: center; margin-top: -30px; font-size: 32px; color: #2C3E50; }
    .subtitle { text-align: center; font-size: 16px; color: #7F8C8D; margin-bottom: 20px; }
    .suggestion-btn button { margin: 4px 8px 4px 0; background-color: #f0f0f0; color: #2C3E50; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='centered-title'>Justify</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Know Your Rights. Clearly. Instantly.</p>", unsafe_allow_html=True)

# ------------------- Session State -------------------
for key in ["chat_history", "chat_chain", "chat_memory", "suggestions", "llm", "glossary", "uploaded_files_count"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history", "chat_memory"] else None

# ------------------- Cached Translator -------------------
@st.cache_resource
def get_translator(model_name):
    return pipeline("translation", model=model_name)

lang_map = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
}

# ✅ Initialize target_lang and translator if missing
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "Spanish"

if "translator" not in st.session_state:
    st.session_state.translator = get_translator(lang_map[st.session_state.target_lang])

# ------------------- Language Selector -------------------
st.sidebar.markdown("### 🌐 Translate Answers")
lang_option = st.sidebar.selectbox(
    "Select Target Language",
    list(lang_map.keys()),
    index=list(lang_map.keys()).index(st.session_state.target_lang),
    key="target_language_selectbox"
)

# ✅ Reload translator when language changes
if st.session_state.target_lang != lang_option:
    st.session_state.target_lang = lang_option
    st.session_state.translator = get_translator(lang_map[lang_option])

# ------------------- Chat Response Function -------------------
def handle_user_input(user_input):
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🤖 Thinking..."):
            result = st.session_state.chat_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_memory
            })
            answer = result["answer"]
            sources = result.get("source_documents", [])

        # ✅ Safe translation
        try:
            translated = st.session_state.translator(answer)[0]['translation_text']
        except Exception:
            translated = "(Translation unavailable)"

        with st.chat_message("assistant"):
            st.markdown(f"*Answer:* {answer}")
            st.markdown(f"*Translated ({st.session_state.target_lang}):* {translated}")
            
            # 🔊 Read Aloud Buttons
            escaped_answer = escape(answer).replace("\n", " ")
            escaped_translated = escape(translated).replace("\n", " ")
            components.html(f"""
                <div style="margin-top: 10px;">
                    <button onclick="var msg = new SpeechSynthesisUtterance('{escaped_answer}'); msg.lang='en-US'; msg.rate=1; window.speechSynthesis.speak(msg);"
                            style="background-color: #3498db; color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer;">
                        Read Original
                    </button>
                    <button onclick="window.speechSynthesis.cancel();"
                            style="background-color: #f44336; color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer;">
                        Stop
                    </button>
                </div>
            """, height=60)

            # Top Sources
            if sources:
                with st.expander("📚 Top Sources"):
                    for i, doc in enumerate(sources[:3]):
                        source_name = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "N/A")
                        content_snippet = doc.page_content.strip().replace('\n', ' ')[:300]

                        st.markdown(f"*Source {i+1}:* {source_name} (Page {page})")
                        st.markdown(f"> {content_snippet}...")
                        st.markdown("---")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.chat_memory.append((user_input, answer))

# ------------------- Upload PDFs -------------------
uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    temp_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_paths.append(tmp_file.name)
        all_docs.extend(load_documents(temp_paths[-1]))

    st.success(f"Uploaded and processed {len(uploaded_files)} file(s).")
    
    # Update chain + llm
    st.session_state.chat_chain, st.session_state.llm = create_temp_qa_chain(all_docs)

    # Generate suggestions only on initial upload
    if not st.session_state.get("uploaded_files_count") == len(uploaded_files):
        with st.spinner("Generating contextual questions..."):
            st.session_state.suggestions = generate_suggestions(all_docs, st.session_state.llm)
            st.session_state.uploaded_files_count = len(uploaded_files)

    # Generate Legal Glossary
    st.session_state.glossary = extract_legal_glossary(all_docs)

    # Sidebar: Legal Glossary
    with st.sidebar.expander("📖 Legal Glossary", expanded=True):
        if st.session_state.glossary:
            for source, glossary_dict in st.session_state.glossary:
                st.markdown(f"**{source}**")
                for category, items in glossary_dict.items():
                    if items:
                        st.markdown(f"**{category}:**")
                        for item in items:
                            st.markdown(f"• {item}")
                st.markdown("---")
        else:
            st.markdown("No glossary items found.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested Questions
    if st.session_state.suggestions:
        st.markdown("*Suggested Questions:*")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggestions):
            col = cols[i % 2]
            if col.button(q, key=f"suggestion_{i}", use_container_width=True):
                handle_user_input(q)

    # Mic Recorder
    with st.expander("Got a question? Say it here."):
        st.markdown("Click the button below to record your voice. The question will be transcribed and asked automatically.")

    audio = mic_recorder(
        start_prompt="Start Speaking",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True
    )

    if audio:
        recognizer = sr.Recognizer()
        audio_bytes = io.BytesIO(audio["bytes"])

        AudioSegment.converter = which("ffmpeg")
        AudioSegment.ffprobe = which("ffprobe")

        try:
            audio_segment = AudioSegment.from_file(audio_bytes, format="webm")
        except Exception as e:
            st.error(f"Could not decode audio: {e}")
            audio_segment = None

        if audio_segment:
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.success("Voice captured successfully!")
                    st.write("What You Said:", f"`{text}`")
                    handle_user_input(text)
                except sr.UnknownValueError:
                    st.error("Could not understand the audio.")
                except sr.RequestError:
                    st.error("Could not reach Google Speech Recognition service.")

    # Manual Text Input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        handle_user_input(user_input)

    # Download Chat History
    if st.session_state.chat_history:
        chat_log = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("Download Chat", data=chat_log, file_name="legal_chat_history.txt", mime="text/plain")

    # Cleanup Temp Files
    for path in temp_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete {path} — {e}")

else:
    st.warning("Upload at least one PDF to begin.")
