import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from typing import List
from ragatouille import RAGPretrainedModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from streamlit_feedback import streamlit_feedback


st.set_page_config(page_title="AIFAL | Beta")
st.title("AIFAL | Beta")

def show_ui(qa, prompt_to_user="Hvordan kan jeg hjelpe deg?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Tenker..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    client = QdrantClient(
        url="https://3d15e88a-6ad1-49db-95b5-dcf7127efc14.europe-west3-0.gcp.cloud.qdrant.io:6333", 
        api_key="Dvc9lxX9C-ZmVCDdkdFGKDHG47x7x-rkCvTiqL_vhfJ7LJf_ks96ig",
    )
    doc_store = Qdrant(
        client=client,
        collection_name="KB_4",
        embeddings=embeddings)
    retriever=doc_store.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 10, 'fetch_k': 50})

    return ContextualCompressionRetriever(
        base_compressor=reranker.as_langchain_document_compressor(k=5), base_retriever=retriever
    )


def get_chain(openai_api_key=None):
    retriever = get_retriever(openai_api_key=openai_api_key)
    chain = create_full_chain(retriever,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key)
        st.subheader("Spør meg om helserelaterte spørsmål")
        show_ui(chain, "Hva vil du vite?")
    else:
        st.stop()


run()
