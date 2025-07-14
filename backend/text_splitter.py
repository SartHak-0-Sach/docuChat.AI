from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    return splitter.split_text(text)