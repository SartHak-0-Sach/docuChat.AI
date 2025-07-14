import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.pdf_loader import extract_text_from_pdf
from backend.text_splitter import split_text
from backend.embedding_store import get_vectorstore
from backend.context_retriever import get_relevant_chunks
from backend.llm_interface import generate_answer

st.set_page_config(page_title="DocuChat AI", layout="wide")
st.title("📄 DocuChat AI — Ask your PDF anything")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

@st.cache_resource
def process_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)
    return get_vectorstore(chunks)

if uploaded_pdf:
    os.makedirs("sample_pdfs", exist_ok=True)
    file_path = os.path.join("sample_pdfs", uploaded_pdf.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_pdf.read())

    with st.spinner("📖 Processing your PDF..."):
        vectorstore = process_pdf(file_path)

    query = st.text_input("Ask something about the document:")

    if query:
        if not query.strip():
            st.warning("❗ Please enter a valid question.")
            st.stop()

        with st.spinner("🔍 Thinking..."):
            relevant_chunks = get_relevant_chunks(query, vectorstore)
            joined_context = "\n".join(relevant_chunks)

            if len(joined_context.strip()) < 20:
                st.info("🤔 Sorry, that doesn't seem relevant to the document.")
            else:
                try:
                    answer = generate_answer(joined_context, query, st.session_state.history)
                    st.success(answer)
                    st.session_state.history.append((query, answer))
                except Exception as e:
                    st.error("⚠️ Couldn't process your query.")
                    st.exception(e)

    with st.expander("🕓 Chat History"):
        for q, a in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")