import streamlit as st
from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="GreenLake AI Chatbot")
st.title("📄 Chat with Document")

# -----------------------------
# LOAD DOCUMENT
# -----------------------------
loader = TextLoader("sample.txt")
documents = loader.load()

# -----------------------------
# SPLIT DOCUMENT
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# -----------------------------
# CREATE EMBEDDINGS + VECTOR DB
# -----------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(temperature=0)

# -----------------------------
# QA CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Ask question from document")

if query:
    result = qa_chain.run(query)
    st.write("### Answer:")
    st.write(result)
