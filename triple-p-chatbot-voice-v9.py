import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import time
import speech_recognition as sr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

# Constants
MAX_MESSAGES = 50
MIN_VOICE_INTERVAL = 2

# Initialize session states
if 'last_voice_request' not in st.session_state:
    st.session_state.last_voice_request = 0

# Check for TTS availability
try:
    import pyttsx3
    TTS_ENABLED = True
except ImportError:
    TTS_ENABLED = False
    st.warning("Text-to-speech package not found. Installing...")
    os.system('pip install pyttsx3')
    try:
        import pyttsx3
        TTS_ENABLED = True
    except ImportError:
        TTS_ENABLED = False
        st.error("Failed to install text-to-speech. Feature will be disabled.")

# Get API key securely
OPENAI_API_KEY = "sk-dRHRWjR3AXWitIp4jLGoxuyL5kKr455G8G3AgiHPAeT3BlbkFJRFF93RBFcsKj68KMz7gkssA7LY7f8vnfWbZLpcra0A"



def check_audio_devices():
    """Check and display available audio devices"""
    try:
        devices = sr.Microphone.list_microphone_names()
        if devices:
            st.sidebar.success(f"Found {len(devices)} microphone(s)")
            st.sidebar.info(f"Default microphone: {devices[0]}")
            return devices
        else:
            st.sidebar.error("No microphones found")
            return []
    except Exception as e:
        st.sidebar.error(f"Error checking audio devices: {str(e)}")
        return []


def initialize_tts():
    """Initialize text-to-speech with error handling"""
    if not TTS_ENABLED:
        return None

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        st.error(f"Error initializing text-to-speech: {str(e)}")
        return None


def get_voice_input(recognizer):
    """Enhanced voice input function with improved error handling and feedback"""
    try:
        # List available microphones
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            st.error("No microphone detected. Please connect a microphone and try again.")
            return None

        # Use selected or default microphone
        mic_index = st.session_state.selected_mic_index
        if mic_index is None:
            st.warning("No microphone selected. Please select a microphone from the sidebar.")
            return None

        with sr.Microphone(device_index=mic_index) as source:
            # Configure recognizer
            recognizer.dynamic_energy_threshold = False
            recognizer.energy_threshold = st.session_state.get('mic_sensitivity', 1000)
            recognizer.pause_threshold = st.session_state.get('pause_threshold', 0.8)

            # Adjust for ambient noise
            st.write("🎤 Adjusting for background noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            st.write("🎤 Listening... (Speak now)")

            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                st.write("Processing your speech...")

                try:
                    text = recognizer.recognize_google(audio)
                    if text:
                        st.success("✓ Processed successfully!")
                        return text
                except sr.UnknownValueError:
                    st.warning("Could not understand audio. Please speak more clearly.")
                except sr.RequestError as e:
                    st.error(f"Could not request results; {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            except sr.WaitTimeoutError:
                st.warning("No speech detected within timeout period. Please try again.")
                return None
            except sr.UnknownValueError:
                st.warning("Could not understand audio. Please speak more clearly.")
                return None

    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        st.info("""Tips:
        - Check if your microphone is properly connected
        - Allow microphone access in your browser
        - Try speaking louder or closer to the microphone
        - Reduce background noise""")
        return None


def speak_text(text):
    """Function for text-to-speech"""
    if not TTS_ENABLED:
        st.warning("Text-to-speech is not available")
        return

    def run_speech():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', st.session_state.get('speech_rate', 175))
            engine.setProperty('volume', st.session_state.get('speech_volume', 1.0))
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            st.error(f"Error during text-to-speech: {str(e)}")

    threading.Thread(target=run_speech).start()

def load_built_in_documents():
    """Load PDF documents with error handling"""
    docs_dir = Path("triple_p_docs")
    docs_dir.mkdir(exist_ok=True)

    all_text = ""
    found_files = False

    try:
        for pdf_path in docs_dir.glob("*.pdf"):
            found_files = True
            try:
                pdf_reader = PdfReader(str(pdf_path))
                text = "".join(page.extract_text() or '' for page in pdf_reader.pages)
                all_text += text + "\n\n"
            except Exception as e:
                st.error(f"Error reading {pdf_path.name}: {str(e)}")
                continue

        if not found_files:
            st.warning("No PDF files found in triple_p_docs directory.")
            return None

        return all_text.strip() if all_text.strip() else None

    except Exception as e:
        st.error(f"Error accessing documents directory: {str(e)}")
        return None


def process_text(text):
    """Process text and create vector store"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None


def handle_voice_input():
    """Handle voice input with rate limiting and feedback"""
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎤 Start Voice Input", key="voice_button"):
            current_time = time.time()
            if current_time - st.session_state.last_voice_request < MIN_VOICE_INTERVAL:
                st.warning("Please wait a moment before trying again")
                return None

            st.session_state.last_voice_request = current_time
            return get_voice_input(st.session_state.recognizer)
    with col2:
        st.info("Click the button and speak your question clearly.")
    return None


# Streamlit app configuration
st.set_page_config(page_title="Triple P Chatbot", page_icon="👪", layout="wide")

# Initialize services in session state
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = initialize_tts()
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()
if 'selected_mic_index' not in st.session_state:
    st.session_state.selected_mic_index = None

# Main header
st.header("Triple P (Positive Parenting Program) Chatbot")

# Initialize vector store
if 'vector_store' not in st.session_state:
    documents_text = load_built_in_documents()
    if documents_text and isinstance(documents_text, str):
        try:
            st.session_state.vector_store = process_text(documents_text)
            st.success("Documents loaded successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.session_state.vector_store = None
    else:
        st.warning("No valid PDF documents found in the triple_p_docs directory.")
        st.session_state.vector_store = None

# Prompt template
PROMPT_TEMPLATE = """
You are an AI assistant trained in the Triple P (Positive Parenting Program) approach.
Use the following pieces of context to answer the question at the end.

Context: {context}

Question: {question}

Please provide a response that:
1. Emphasizes Triple P core principles
2. Encourages positive parenting techniques
3. Promotes self-regulation
4. Suggests age-appropriate strategies
5. Encourages adaptation to family needs
6. Avoids criticism
7. Reminds that change takes time
8. dont be too long

Answer:
"""

TRIPLE_P_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Input interface
col1, col2 = st.columns([3, 1])
with col1:
    input_method = st.radio("Choose input method:", ["Text", "Voice"])
with col2:
    enable_tts = st.checkbox(
        "Enable Voice Response",
        value=False,
        disabled=not TTS_ENABLED,
        help="Text-to-speech must be properly installed to use this feature"
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_question = None

if input_method == "Voice":
    user_question = handle_voice_input()
else:
    user_question = st.chat_input("Ask a question about parenting using Triple P strategies")

# Process question and generate response
if user_question and st.session_state.vector_store:
    if len(st.session_state.messages) >= MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-(MAX_MESSAGES - 1):]

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            docs = st.session_state.vector_store.similarity_search(user_question)

            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=1500,
                model_name="gpt-3.5-turbo"
            )

            chain = load_qa_chain(
                llm=llm,
                chain_type="stuff",
                prompt=TRIPLE_P_PROMPT
            )

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            full_response = response.get("output_text", "I apologize, but I couldn't generate a response.")

            # Simulate typing effect
            words = full_response.split()
            displayed_text = ""
            for word in words:
                displayed_text += word + " "
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)

            if enable_tts and TTS_ENABLED:
                with st.spinner("Converting response to speech..."):
                    speak_text(full_response)


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = "I apologize, but I encountered an error while processing your question."
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif user_question and not st.session_state.vector_store:
    st.error("Please add PDF documents to the triple_p_docs directory before asking questions.")

# Sidebar settings
with st.sidebar:
    st.subheader("Voice Settings")

    # Audio device information
    if st.checkbox("Show Audio Device Info"):
        mic_list = check_audio_devices()
        if mic_list:
            selected_mic = st.selectbox("Select Microphone", mic_list)
            if st.button("Apply Microphone"):
                st.session_state.selected_mic_index = mic_list.index(selected_mic)
                st.success(f"Microphone set to: {selected_mic}")

    # Voice configuration
    if st.checkbox("Show Voice Configuration"):
        if TTS_ENABLED and st.session_state.tts_engine is not None:
            speech_rate = st.slider("Speech Rate", 100, 300, 175)
            speech_volume = st.slider("Speech Volume", 0.0, 1.0, 1.0)
            mic_sensitivity = st.slider("Microphone Sensitivity", 100, 4000, 1000)
            pause_threshold = st.slider("Pause Threshold", 0.5, 3.0, 0.8)

            if st.button("Apply Voice Settings"):
                try:
                    st.session_state.tts_engine.setProperty('rate', speech_rate)
                    st.session_state.tts_engine.setProperty('volume', speech_volume)
                    st.session_state.mic_sensitivity = mic_sensitivity
                    st.session_state.pause_threshold = pause_threshold
                    st.success("Voice settings updated!")
                except Exception as e:
                    st.error(f"Error updating voice settings: {str(e)}")
        else:
            st.warning("Text-to-speech functionality is not available")

    # Debug information
    if st.checkbox("Show Debug Info"):
        st.write("Audio Settings:")
        st.write(f"Energy Threshold: {st.session_state.recognizer.energy_threshold}")
        st.write(f"Dynamic Energy: {st.session_state.recognizer.dynamic_energy_threshold}")
        st.write(f"Pause Threshold: {st.session_state.recognizer.pause_threshold}")
        st.write(f"Selected Mic Index: {st.session_state.selected_mic_index}")

# About section and resources in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About Triple P")
st.sidebar.info(
    "The Triple P (Positive Parenting Program) is a parenting and family support system "
    "designed to prevent and treat behavioral and emotional problems in children and teenagers. "
    "It aims to equip parents with the skills and confidence they need to be self-sufficient "
    "and to manage family issues without ongoing support."
)

# Troubleshooting section
st.sidebar.markdown("### Troubleshooting Voice Input")
st.sidebar.info(
    """If voice input is not working:
    1. Check microphone connection
    2. Allow browser microphone access
    3. Speak clearly and at a normal volume
    4. Reduce background noise
    5. Try adjusting microphone sensitivity
    6. Select a different microphone if available
    7. Restart the application if issues persist
    """
)

# Resource links
st.sidebar.markdown("### Useful Resources")
st.sidebar.markdown("- [Triple P Official Website](https://www.triplep.net)")
st.sidebar.markdown("- [Triple P Online](https://www.triplep-parenting.com)")
st.sidebar.markdown("- [Research on Triple P](https://pfsc.evidence.psy.uq.edu.au/)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>Version 1.0.0 | Made with ❤️ for parents</small>
    """,
    unsafe_allow_html=True
)
