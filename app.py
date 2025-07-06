import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"
main_placeholder = st.empty()

# Set up Gemini Flash model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    main_placeholder.text("Text Splitting...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Use Fake Embeddings to simulate embedding step
    embeddings = FakeEmbeddings(size=768)
    vectorstore = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore.serialize_to_bytes()

    main_placeholder.text("Building Vector Index...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

# Input query
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)

        vectorstore = FAISS.deserialize_from_bytes(
            embeddings=FakeEmbeddings(size=768),
            serialized=pkl,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
