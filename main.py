# imports

import os
from dotenv import load_dotenv
import streamlit as st
import re
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot (MongoDB + GPT-4o)")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not MONGO_URI:
    st.error("MONGODB URI not found in environment variables")
    st.stop()

# create MongoDB client
client = MongoClient(MONGO_URI)

# ping MongoDB to verify secure connection
try:
    client.admin.command("ping")
    print("✅ Securely connected to MongoDB Atlas")
    st.success("✅ Securely connected to MongoDB Atlas")
except Exception as e:
    print("❌ MongoDB connection failed")
    st.error("❌ MongoDB connection failed")
    st.exception(e)
    st.stop()

collection = client[DB_NAME][COLLECTION_NAME]

# Embeddings + LLM
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)


# load and clean
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"Page \d+", "", text)
    return text.strip()


def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
