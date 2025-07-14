from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb.config

def get_vectorstore(chunks, persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    chroma_settings = chromadb.config.Settings(chroma_api_impl="chromadb.api.local.LocalAPI")
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        client_settings=chroma_settings,
    )
    return vectorstore