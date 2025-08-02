import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Langsmith for monitoring
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Setup LangChain components
llm = OllamaLLM(model='gemma3:1b')
output_parser = StrOutputParser()
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Please response to the user queries"),
    ("user", "Question:{question}")
])

def get_chain():
    return prompt | llm | output_parser

# --- Frontend UI: Render BILLA Bot yellow badge ---
def render_bot_badge():
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="
                min-width: 60px;
                height: 60px;
                background: #FFD600;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                color: #222;
                font-size: 1.2em;
                border: 2px solid #f9c900;
                margin-right: 12px;"
            >
                BILLA
            </div>
            <span style="font-weight: bold; font-size: 1.3em; color: #FFD600;"></span>
        </div>
        """, unsafe_allow_html=True)

def inject_custom_css_left_billa():
    st.markdown("""
        <style>
        /* Remove default padding/margin to left-align everything */
        .main, .block-container {
            padding-left: 0 !important;
            margin-left: 0 !important;
            max-width: 100vw !important;
        }
        </style>
    """, unsafe_allow_html=True)




# --- Frontend UI: Render the chat history ---
def render_chat_history():
    st.subheader("💬 Chat History")
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {exchange['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"**BILLA:** {exchange['assistant']}")

    st.markdown("---")
    if st.button("🔁 Reset Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.experimental_rerun()

def main():
    st.set_page_config(page_title="BILLA ChatBot", page_icon="🤖")

    # Inject CSS for left align and font size
    #inject_custom_css_left_billa()

    # Render top badge
    render_bot_badge()

    st.title("🤖 My First ChatBot Application")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    chain = get_chain()

    # Input form
    with st.form("input_form", clear_on_submit=True):
        input_col1, input_col2 = st.columns([5, 1])
        with input_col1:
            user_input = st.text_input("Ask BILLA something...", key="input_text", label_visibility="collapsed")
        with input_col2:
            submitted = st.form_submit_button("Send", use_container_width=True)

    # On input submit
    if submitted and user_input.strip():
        response = chain.invoke({"question": user_input})
        st.session_state.chat_history.append({"user": user_input, "assistant": response})

    # Render chat history UI
    render_chat_history()

    # Footer
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.85em; color: gray; margin-top: 30px;">
        Powered by <b>Govardhan Billa</b>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
