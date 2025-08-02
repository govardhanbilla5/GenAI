import streamlit as st
import requests


def get_response_ollama(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json = {"input":{"topic":input_text}}
        )
    return response.json()['output']

def get_response_ollama2(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json = {"input":{"topic":input_text}}
        )
    return response.json()['output']


if __name__ == "__main__":
    st.title("Web app Demo with Langchain, llama and FastAPI")
    input_text = st.text_input("write an essay on")
    input_text1 = st.text_input("write a poem on")

    
    if input_text:
        st.write(get_response_ollama(input_text))
    if input_text1:
        st.write(get_response_ollama(input_text1))
    
    