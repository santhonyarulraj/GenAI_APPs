import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Get API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

st.title("🏠 House Rental FAQ - RAG App")

uploaded_file = st.file_uploader("Upload FAQ .txt file", type="txt")

if uploaded_file:

    text = uploaded_file.read().decode("utf-8")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Strict Prompt
    prompt_template = """
You are a helpful assistant answering ONLY from the provided FAQ document.

If the answer is not found in the FAQ document, respond exactly with:
"Information not present in the FAQ document."

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    user_query = st.text_input("Ask your question about the house:")

    if user_query:
        response = qa_chain.run(user_query)
        st.write("### Answer:")
        st.write(response)
