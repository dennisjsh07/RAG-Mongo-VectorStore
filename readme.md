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

![App Screenshot](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftchtfwyspxpuwwfzwtgp.png)

