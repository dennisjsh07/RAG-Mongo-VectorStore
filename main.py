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
        embedding = embeddings.embed_query(chunk.page_content)

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
            "$vectorSearch": {
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


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not found, say "Not found in the document."

Context:
{context_text}

Question:
{query}
"""
    return llm.stream(prompt)


# ----------- UI ----------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("📥 Ingest PDF"):
        with st.spinner("Ingesting PDF..."):
            ingest_pdf(uploaded_file.name)
        st.success("PDF ingested successfully!")

st.divider()

query = st.text_input("Ask a question about the PDF")

submit = st.button("🔍 Submit")

if submit and query:
    results = vector_search(query)

    contexts = [r["text"] for r in results]

    st.subheader("🤖 Answer")
    answer_container = st.empty()

    full_answer = ""
    for chunk in generate_answer(query, contexts):
        full_answer += chunk.content
        answer_container.markdown(full_answer)

"""
    st.subheader("📚 Sources")
    for r in results:
        st.caption(f"Page {r['metadata'].get('page')} — Score: {round(r['score'], 3)}")

"""
