import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

# import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI, GooglePalm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

load_dotenv()

with st.sidebar:
    st.title("PdfAI")
    st.markdown(
        """
        ## About
        This app is an LLL-powered pdf chatbot but uisng
        - Streamlit
        - Langchain
        - OpenAI
    """
    )

    add_vertical_space(5)
    st.write("Made by Abdullah Saihan Taki!")


def main():
    st.header("Chat with PDF!")

    pdf = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt"],
    )

    if pdf is not None:
        file_name = pdf.name[:-4]
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        embeddings = GooglePalmEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        if os.path.exists(f"{file_name}.dat"):
            VectorStore = FAISS.load_local(
                f"{file_name}.dat", embeddings, allow_dangerous_deserialization=True
            )
            st.write("Embeddings Loaded from the Disk")
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{file_name}.dat")
            st.write("Embeddings computation completed!")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask questions about your PDF file:")

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            docs = VectorStore.similarity_search(query=prompt, k=3)
            llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"])
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=prompt)
            with st.chat_message("assistant"):
                st.write(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
