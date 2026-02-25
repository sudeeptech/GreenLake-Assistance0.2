import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="GreenLake AI Chatbot")
st.title("📄 Chat with Document (RAG)")

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
# LOCAL EMBEDDINGS (FREE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# VECTOR STORE
# -----------------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# -----------------------------
# GROQ LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# -----------------------------
# RETRIEVAL QA
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
