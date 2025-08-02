from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi import FastAPI
import uvicorn

app=FastAPI(
    title='LangChain Server',
    version='1.0',
    description='A simple API Server for LLMs'
)


# Load Model
llm = OllamaLLM(
    model='gemma3:1b'
)

add_routes(
    app,
    llm,
    path="/openai"
)
#Like this you can load different llm models like openai, grok, ..


# Create Prompts 
prompt1 = ChatPromptTemplate.from_template(
    "Write an essay about on {topic} with 50 lines"
)


# Create Prompts 
prompt2 = ChatPromptTemplate.from_template(
    "Write an essay a poem on {topic} with 50 lines"
)


# Add routes
add_routes(
    app,
    prompt1|llm,
    path='/essay'
)
add_routes(
    app,
    prompt2|llm,
    path='/poem'
)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)