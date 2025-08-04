
# #Simple Caht Bot
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# #Load Model
# llm = OllamaLLM(model='gemma3:1b')
# output_parser = StrOutputParser()

# # Design Prompt
# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful assistant, please response to the user queries"),
#     ("user","Question:{question}")
# ])


# chain = prompt|llm|output_parser

# while True:
#     inp = input("You:")
#     response = chain.invoke({"question":inp})
#     print("Bot:", response)


# #Simple Chat Bot using Streamlit Web App
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# #Load Model
# llm = OllamaLLM(model='gemma3:1b')
# output_parser = StrOutputParser()

# # Design Prompt
# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful assistant, please response to the user queries"),
#     ("user","Question:{question}")
# ])


# chain = prompt|llm|output_parser

# st.title("My First ChatBot App -> ╰(*°▽°*)╯")
# input_text = st.text_input("You:")
# if input_text:
#     st.write("Bot:", chain.invoke({"question":input_text}))


# #Simple Chat Bot using Streamlit Web App to show history in web App
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# #Load Model
# llm = OllamaLLM(model='gemma3:1b')
# output_parser = StrOutputParser()

# # Design Prompt
# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful assistant, please response to the user queries"),
#     ("user","Question:{question}")
# ])


# chain = prompt|llm|output_parser

# st.title("My First ChatBot App -> ╰(*°▽°*)╯")

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []



# input_text = st.text_input("You:")
# if input_text:
#     response = chain.invoke({"question":input_text})
#     st.session_state.chat_history.append({"user":input_text, "assistant":response})


# st.subheader("Chat History")
# for exchange in st.session_state.chat_history:
#     st.markdown(f"**You**:{exchange['user']}")
#     st.markdown(f"**Bot**:{exchange['assistant']}")

# if st.button("Reset Chat"):
#     st.session_state.chat_history = []



# RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#For cahin
from langchain.chains.combine_documents import create_stuff_documents_chain
#For retriever
from langchain.chains import create_retrieval_chain



# Load Documents
loader = PyMuPDFLoader('attention.pdf')
docs = loader.load()
# Convert document into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
# print(documents)

# Convert it into Embeddings and store it in database
embeddings = OllamaEmbeddings(model='all-minilm')
db = Chroma.from_documents(documents, embeddings)
# print(db)

# # Ask question and retrieve data from document with similarity check
# query = "Who are the authors of attention is all you need?"
# retriever = db.similarity_search(query)
# print(retriever[0].page_content)


# Retrieve answer from user query using LLM
llm = OllamaLLM(model='gemma3:1b')


prompt = ChatPromptTemplate.from_template("""
Answe the following question based only on the provided context.
Think step by step before providing the detailed answer.
If the question is not matched in context then give answer as not available the data.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")



# chain
document_chain = create_stuff_documents_chain(llm, prompt)
#retriever
retriever = db.as_retriever()

#Retriver Chain
retrival_chain = create_retrieval_chain(retriever, document_chain)

#Ask Query
# input_text = "what is Newtons third law of motion"
input_text = "what is self attention"
response = retrival_chain.invoke({"input":input_text})
print(response['answer'])
