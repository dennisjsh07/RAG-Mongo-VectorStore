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
VECTOR_INDEX = os.getenv("VECTOR_INDEX")

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

    for d in docs:
        d.page_content = clean_text(d.page_content)
        print(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)

        collection.insert_one(
            {
                "text": chunk.page_content,
                "embedding": embedding,
                "metaData": {
                    "page": chunk.metadata.get("page"),
                    "source": file_path,
                    "chunk_id": i,
                },
            }
        )


def vector_search(query, k=4):
    query_embedding = embeddings.embed_query(query)

    pipeline = [
        {
            "$vector_search": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))
