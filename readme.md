## Building a document reader chatbot 

This project explains the architecture and Implementation of RAG with mongodb vector stores.

### Architecture
PDF Upload (Streamlit)
 → Load & Clean
 → Chunk
 → Embed
 → Store in MongoDB Vector Index

User Question
 → Embed Query
 → MongoDB Vector Search
 → Context
 → GPT-4o (grounded prompt)
 → Streamed Answer


