from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st

# Load Document
loader = PyMuPDFLoader('attention.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model='all-minilm')
db = FAISS.from_documents(documents, embeddings)

#Design Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the following questions based only on the provided context.
Think step by step before providing the answer.
If the question is not matched in context then give answer as not available the data.
I will tip yoi $1000 if user finds the helpful answer.
<context>
{context}
</context>
Question: {input}""")

# Load LLM
llm = OllamaLLM(model="gemma3:1b")

# Create Document Chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create Retrieval Chain
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


st.title("My First RAG Application ╰(*°▽°*)╯")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]

# Q&A
input_text = st.text_input("You")
if input_text:
    response = retrieval_chain.invoke({"input":input_text})
    st.session_state.chat_history.append({"user":input_text, "assistant":response['answer']})

st.subheader("Chat History")

for Exchange in st.session_state.chat_history:
    st.markdown(f"**You:** {Exchange['user']}")
    st.markdown(f"**Bot:** {Exchange['assistant']}")

if st.button('Reset Chat'):
    st.session_state.chat_history = []