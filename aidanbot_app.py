# Lady Gurumi’s AidanBot Web App
# A cute LangChain + Streamlit chatbot powered by our support_qa.txt file

# === 1.- Load Environment and Libraries ===

import os
from dotenv import load_dotenv
load_dotenv()  # Load our OpenAI API key securely from a .env file

import streamlit as st
from PIL import Image # For our banner
from streamlit.components.v1 import html  # Allows inserting raw HTML + JavaScript

# LangChain modules to load LLM, embed docs, chunk them, and answer questions
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === 2-. Streamlit App Configuration ===

st.set_page_config(page_title="Lady Gurumi's AidanBot", page_icon="🧸") # Browser title

# Customize the look of the input box
st.markdown("""
    <style>
    input[type="text"] {
        background-color: #fff8ff;
        color: #333333;
        padding: 0.6em;
        border-radius: 12px;
        border: 1px solid #f0e6f6;
        font-size: 16px;
    }
    ::placeholder {
        color: #b288cc;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)


# === 3.- UI Banner and Title ===

banner_image = Image.open("lady_gurumi_banner.png")  # Our cute graphic banner
st.image(banner_image, use_container_width=True)

# Main title in black
st.markdown(
    "<h3 style='text-align: center; color: #000000;'>✨ Lady Gurumi’s Support Agent ✨</h3>",
    unsafe_allow_html=True
)


# === 4.- Load and Configure LangChain Components ===

from chromadb.config import Settings

@st.cache_resource
def load_chain():
    # Load the Q&A document
    loader = TextLoader("support_qa.txt", encoding="utf-8")
    docs = loader.load()

    # Split into 512-token chunks with 50-token overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Embed the chunks using MiniLM model
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None  # In-memory only
    )
    
    # Creates in-memory vector store using ChromaDB
    vectorstore = Chroma.from_documents(
    split_docs,
    embedding=embedding_model,
    client_settings=client_settings
)

    
    # Create the in-memory vector store
    vectorstore = Chroma.from_documents(
        split_docs,
        embedding=embedding_model,
        collection_name="aidanbot"  # Optional, but helps keep it organized
    )

    # Define AidanBot’s tone
    cute_prompt = PromptTemplate.from_template(
        """You are Lady Gurumi’s adorable support assistant, known as AidanBot.
Always answer in a cute, witty, and nerdy tone, using emojis or cozy language if appropriate.
Answer this customer question using only the provided context.
If you're not sure, say something funny and kind.
Only mention Moni the Plushie Wizard if the question cannot be answered from the context or if it's something personal, subjective, or requires a human decision.
Remember that Moni is a woman and every time you talk about what she does or how to contact her you have to use third person in grammar.
Context:
{context}

Question:
{question}

Answer:"""
    )

    # Create the LangChain RetrievalQA system
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4),
        retriever=vectorstore.as_retriever(score_threshold=0.6),
        chain_type_kwargs={"prompt": cute_prompt, "document_variable_name": "context"}
    )
qa_chain = load_chain()


# === 5.- Initialize Memory to Store Chat Messages ===

if "messages" not in st.session_state:
    st.session_state.messages = []  # List of Q&A pairs for display


# === 6.- Define What Happens When User Submits a Question ===

def submit_message():
    # Add user message + AidanBot response to memory
    st.session_state.messages.append({
        "question": st.session_state.user_input,
        "answer": qa_chain.invoke(st.session_state.user_input)["answer"]
    })
    st.session_state.user_input = ""  # Clear box after submitting


# === 7.- Display Scrollable Chat Window with Messages ===

if st.session_state.messages:
    chat_html = """
    <html>
    <head>
    <style>
        .scrollbox {
            height: 200px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #f0e6f6;
            border-radius: 12px;
            background-color: #fff8ff;
            font-family: sans-serif;
            font-size: 16px;
        }
        .scrollbox::-webkit-scrollbar {
            width: 6px;
        }
        .scrollbox::-webkit-scrollbar-thumb {
            background-color: #d9b9f5;
            border-radius: 6px;
        }
    </style>
    </head>
    <body>
    <div class='scrollbox' id='chatbox'>
    """

    # Print each previous Q&A inside the scrollbox
    for entry in st.session_state.messages:
        chat_html += f"<p>🧶 <b>You:</b> {entry['question']}</p>"
        chat_html += f"<p>🧸 <b>AidanBot:</b> {entry['answer']}</p>"
        chat_html += "<hr style='margin: 6px 0;'>"

    # Scroll to bottom automatically
    chat_html += """
    </div>
    <script>
        var chatbox = document.getElementById("chatbox");
        chatbox.scrollTop = chatbox.scrollHeight;
    </script>
    </body>
    </html>
    """

    html(chat_html, height=200)


# === 8.- User Input Box ===

st.text_input(
    label="",
    key="user_input",
    placeholder="Ask me anything about returns, shipping, plushies, or emotional support robots...",
    on_change=submit_message
)
