import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# -----------------------
# LOAD ENV
# -----------------------
load_dotenv()

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="GreenLake Assistant")
st.title("📄 Chat with Document (RAG)")

# -----------------------
# LOAD DOCUMENT
# -----------------------
loader = TextLoader("sample.txt")
documents = loader.load()

# -----------------------
# SPLIT TEXT
# -----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# -----------------------
# EMBEDDINGS (LOCAL)
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# VECTOR STORE
# -----------------------
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------
# LLM (GROQ)
# -----------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# -----------------------
# PROMPT (STRICT RAG)
# -----------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say: "I don't know".

Context:
{context}

Question:
{input}
""")

# -----------------------
# CREATE CHAINS
# -----------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -----------------------
# USER INPUT
# -----------------------
query = st.text_input("Ask a question from the document")

if query:
    response = retrieval_chain.invoke({"input": query})
    st.write("### Answer:")
    st.write(response["answer"])
