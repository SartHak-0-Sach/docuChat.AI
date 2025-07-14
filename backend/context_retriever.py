from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def get_relevant_chunks(query, vectorstore, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

