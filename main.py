# imports

import os
from dotenv import load_dotenv
import streamlit as st
import re
from pymongo import MongoClient
from langchain_community import document_loaders
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


