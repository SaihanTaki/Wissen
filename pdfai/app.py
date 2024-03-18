from fileinput import filename
import os
import shutil
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from handlefiles import File2Text
from langchain_community.llms.google_palm import GooglePalm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


def get_llm(temparature=0.1):
    # Workaround on first initialization of googl palm raising NotImplemetedError
    try:
        llm = GooglePalm(
            google_api_key=os.environ["GOOGLE_API_KEY"], temparature=temparature
        )
    except NotImplementedError:
        llm = GooglePalm(
            google_api_key=os.environ["GOOGLE_API_KEY"], temparature=temparature
        )
    return llm


def get_file_text(files):
    text = ""
    for file in files:
        text += f"{file.name}" + "\n" + File2Text(file)()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_vectorstore(text_chunks):

    embeddings = GooglePalmEmbeddings()

    vector_db_path = "vectorstore"
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)
    db_name = "vectorstore.dat"
    db_path = os.path.join(vector_db_path, db_name)

    if os.path.exists(db_path):
        vector_db = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        st.write("Embeddings Loaded from the Disk")
    else:
        vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_db.save_local(db_path)
        st.write("Embeddings computation completed!")

    return vector_db


def get_conversation_chain(vectorstore=None):
    llm = get_llm(temparature=0.0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Human",
        ai_prefix="AI Aassistant",
        return_messages=True,
    )
    if vectorstore:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    else:
        template = """The following is a friendly conversation between a human and an AI.
                The AI is talkative and provides lots of specific details from its context.
                If the AI does not know the answer to a question, it truthfully says it does\
                not know.
                If users ask abot questions related to any document say the following
                - It looks like you haven't uploaded any document yet.
                Current conversation:
                {chat_history}
                Human: {question}
                AI Assistant:"""
        PROMPT = PromptTemplate(
            input_variables=["chat_history", "question"], template=template
        )

        conversation_chain = ConversationChain(
            prompt=PROMPT,
            llm=llm,
            verbose=False,
            memory=memory,
            input_key="question",
            output_key="answer",
        )
    return conversation_chain


def main():

    load_dotenv()

    st.set_page_config(page_title="Wissen", page_icon=":books:")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()

    # Sidebar
    with st.sidebar:
        st.title("Wissen")

        st.markdown(
            """
            ##### `Wissen` is the German of `Knowledge`
            
            ## About
            This app is an LLL-powered multi-document chatbot uisng
            - Streamlit
            - Langchain
            - GooglePalm

           
            """
        )
        st.subheader("Your Documents")
        files = st.file_uploader(
            "Upload a pdf, docx, or txt file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if st.button("Process"):
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            with st.spinner("Processing"):
                if files:
                    # get pdf text
                    raw_text = get_file_text(files)
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.session_state.conversation = get_conversation_chain()

        add_vertical_space(5)
        st.write("Made by Abdullah Saihan Taki!")

    st.header("Wissen: Chat with multiple Documents")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask questions about your document")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                response = st.session_state.conversation({"question": prompt})["answer"]
                st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
