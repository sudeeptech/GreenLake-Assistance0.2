import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY missing in .env file")
    st.stop()

# -------------------------
# STREAMLIT PAGE
# -------------------------
st.set_page_config(
    page_title="GreenLake Assist",
    page_icon="🤖",
    layout="centered",
)

st.title("💬 GreenLake Assist (RAG)")

if st.button("🔄 Reload Document"):
    st.cache_resource.clear()
    st.rerun()

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# LLM INIT (GROQ)
# -------------------------
llm = ChatGroq(
    model="llama3-8b-8192",   # stable model
    temperature=0
)

# -------------------------
# RAG SETUP
# -------------------------
@st.cache_resource
def setup_rag():

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    loader = TextLoader("sample.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_rag()

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # retrieve context
    docs = retriever.invoke(user_prompt)
    context = "\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
You are an internal company support assistant.

Follow these rules strictly:
1. Answer ONLY from the provided context.
2. If answer not found → say "I don't know".
3. Use simple professional language.
4. Use bullet points when helpful.

Context:
{context}

User Question:
{user_prompt}

Helpful Answer:
"""

    try:
        response = llm.invoke(rag_prompt)
        assistant_response = response.content
    except Exception as e:
        assistant_response = f"Error: {e}"

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
