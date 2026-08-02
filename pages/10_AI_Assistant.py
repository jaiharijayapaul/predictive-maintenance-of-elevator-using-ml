import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.chatbot_engine import ElevatorChatbot

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

def load_css():
    css_path = PROJECT_ROOT / "assets" / "css" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🤖 AI Project Assistant")
st.markdown("""
Welcome to the **Elevator Predictive Maintenance AI Assistant**! 
I am trained on this project's documentation and architecture. Ask me anything about how the models work, the feature engineering, or general machine learning concepts.
""")

# Attempt to get API key from secrets
try:
    api_key = st.secrets.get("GROQ_API_KEY", None)
except FileNotFoundError:
    api_key = None

with st.sidebar:
    st.header("⚙️ Configuration")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", help="Get your free key from console.groq.com/keys")
        if not api_key:
            st.warning("Please provide a Groq API Key.")
    else:
        st.success("✅ Groq API Key loaded!")

    with st.expander("🔑 How to get a free Groq API key"):
        st.markdown("""
**Step 1:** Go to [console.groq.com/keys](https://console.groq.com/keys)

**Step 2:** Sign in with Google or GitHub

**Step 3:** Click **Create API Key**, give it a name

**Step 4:** Copy the key (starts with `gsk_...`)

**Step 5:** Paste it in `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_..."
```
**Free tier:** 14,400 requests/day ⚡
        """)
    
    st.header("💡 Quick Questions")
    if st.button("How does the Random Forest model work?"):
        st.session_state.preset_question = "How does the Random Forest model work in this project?"
    if st.button("Explain the Motor Stress Index"):
        st.session_state.preset_question = "Can you explain how the Motor Stress Index is calculated and used?"
    if st.button("Why was SMOTE used?"):
        st.session_state.preset_question = "Why did you use SMOTE for balancing the dataset in this project?"
        
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        if "chatbot" in st.session_state:
            del st.session_state.chatbot
        st.rerun()

if api_key:
    # Initialize the chatbot
    if "chatbot" not in st.session_state:
        try:
            with st.spinner("Connecting to AI model..."):
                st.session_state.chatbot = ElevatorChatbot(api_key=api_key)
                st.session_state.chat_session = st.session_state.chatbot.create_chat_session()
        except RuntimeError as e:
            st.error("❌ Could not connect to any Gemini model.")
            st.warning(
                "**Your API key appears to be invalid or quota-exhausted.**\n\n"
                "Keys starting with `AQ.` are **not** Google AI Studio developer keys.\n\n"
                "Please get a free key from 👉 https://aistudio.google.com/app/apikey "
                "(it will start with `AIzaSy...`)."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {e}")
            st.stop()

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI assistant for this project. How can I help you today?"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle input
    prompt = st.chat_input("Ask me about the project...")
    
    if "preset_question" in st.session_state and st.session_state.preset_question:
        prompt = st.session_state.preset_question
        st.session_state.preset_question = None
        
    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response with streaming
        with st.chat_message("assistant"):
            try:
                # In case the session expired or is missing
                if "chat_session" not in st.session_state:
                    st.session_state.chat_session = st.session_state.chatbot.create_chat_session()

                stream = st.session_state.chatbot.send_message_stream(
                    st.session_state.chat_session, prompt
                )

                def stream_text():
                    for chunk in stream:
                        if chunk:   # chunk is already a plain string from Groq
                            yield chunk

                full_response = st.write_stream(stream_text())
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                err_str = str(e)
                if '429' in err_str:
                    st.warning(
                        "⚠️ **Quota exceeded.** Your API key has hit its rate limit.\n\n"
                        "Wait a minute and try again, or get a new key from "
                        "[aistudio.google.com](https://aistudio.google.com/app/apikey)."
                    )
                elif '401' in err_str or '403' in err_str:
                    st.error("❌ **Invalid API key.** Please check your key in `.streamlit/secrets.toml`.")
                else:
                    st.error(f"❌ An error occurred: {err_str[:300]}")
else:
    st.info("👈 Please enter your Groq API Key in the sidebar or set it in `.streamlit/secrets.toml` to start chatting.")
