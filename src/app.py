import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from extract_queries import extract_query
from web_scraper import fetch_web_pages
from db_operations import get_embedding_function, search_in_file
from prompt_generator import generate_prompt
import asyncio
import ollama
import tempfile
import time
import os

st.set_page_config(page_title="WEB RAG ENGINE", page_icon="🤖")
st.title("WEB RAG ENGINE")


def chunk_generator(llm, query):
    for chunk in llm.stream(query):
        yield chunk


with st.sidebar:
    llm_model = st.selectbox(
        label="Select llm model",
        options=[
            model.model
            for model in ollama.list().models
            if model.model != "nomic-embed-text:latest"
        ],
    )
    search_engine = st.selectbox(
        label="Select search engine", options=["google", "duckduckgo"]
    )
    n_results = st.number_input(
        label="Select number of web results", min_value=1, max_value=8, value=1
    )
    uploaded_file = st.file_uploader("Upload file (PDF/TXT)", type=["pdf", "txt"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web or uploaded files. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if usr_msg := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": usr_msg})
    st.chat_message("user").write(usr_msg)

    with st.chat_message("assistant"):
        with st.spinner("extracting queries..."):
            queries = extract_queries(usr_msg, model=llm_model)

        sources = ""
        prompt = ""
        embedding_function = get_embedding_function()

        # 1. Try searching in the uploaded file first (if exists)
        file_result = None
        if uploaded_file is not None:
            with st.spinner("Searching uploaded file..."):
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name

                start_time = time.time()
                file_result = search_in_file(
                    queries, file_path, embedding_function, timeout=30
                )
                elapsed_time = time.time() - start_time

                if file_result:
                    prompt, sources, score = file_result

                    if score > 0.7:
                        st.success("Answer found in uploaded file!")
                    else:
                        file_result = None
                        st.warning(
                            "No relevant info found in file. Switching to web search..."
                        )
                else:
                    st.warning(
                        "No relevant info found in file within 30 seconds. Switching to web search..."
                    )

        # 2. If file search failed or no file uploaded, go to web
        if not file_result:
            with st.spinner("searching on the web..."):
                asyncio.run(fetch_web_pages(queries, n_results, provider=search_engine))

            with st.spinner("extracting info from webpages..."):
                prompt, sources = generate_prompt(usr_msg, embedding_function)

        with st.spinner("generating response..."):
            llm = ChatOllama(model=llm_model, stream=True)
            stream_data = chunk_generator(llm, prompt)
            st.write_stream(stream_data)
            # st.write(sources)
