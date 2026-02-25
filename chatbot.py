import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY missing in .env file")
    st.stop()

# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(page_title="GreenLake Assist", page_icon="🤖")
st.title("💬 GreenLake Assist — Chat with Document")

# -----------------------------
# RAG SETUP
# -----------------------------
@st.cache_resource
def setup_rag():
    loader = TextLoader("sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_rag()

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192", temperature=0)

# -----------------------------
# Prompt & Chains
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an internal company assistant.

Answer ONLY from the provided context.
If answer is not available → say "I don't know".
Use simple professional language.
Use bullet points if helpful.

Context:
{context}

Question:
{input}

Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -----------------------------
# Chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User input
# -----------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    try:
        response = retrieval_chain.invoke({"input": user_prompt})
        assistant_response = response["answer"]
    except Exception as e:
        assistant_response = f"Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
