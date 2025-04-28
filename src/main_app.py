import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from extract_queries import extract_queries
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
        label="Select LLM model",
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
if usr_msg := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": usr_msg})

    with st.chat_message("assistant"):
        with st.spinner("Extracting queries from your question..."):
            queries = extract_queries(usr_msg, model=llm_model)

        sources = ""
        prompt = ""
        embedding_function = get_embedding_function()

        file_result = None
        source_type = "🌐 Web"  # Default to Web

        if uploaded_file is not None:
            with st.spinner("Searching uploaded file..."):
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name

                start_time = time.time()
                file_result = search_in_file(queries, file_path, embedding_function, timeout=30)
                elapsed_time = time.time() - start_time

                if file_result:
                    prompt, sources, score = file_result
                    if score > 0.8:
                        st.success("✅ Answer found in uploaded file!")
                        source_type = "🗎 File"
                    else:
                        file_result = None
                        st.warning("⚠️ No relevant info found in file. Switching to web search...")
                else:
                    st.warning("⚠️ No relevant info found in file within 30 seconds. Switching to web search...")

        if not file_result:
            with st.spinner("Searching on the web..."):
                asyncio.run(fetch_web_pages(queries, n_results, provider=search_engine))

            with st.spinner("Extracting information from web pages..."):
                prompt, sources = generate_prompt(usr_msg, embedding_function)

        with st.spinner("Generating response using LLM..."):
            llm = ChatOllama(model=llm_model, stream=True)
            answer = ""

            for chunk in chunk_generator(llm, prompt):
                answer += chunk.content

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "source": source_type  # Save source type too
            })

# Render previous conversations with expanders
total_pairs = len(st.session_state.messages) // 2  # number of user-assistant pairs

for idx in range(0, len(st.session_state.messages), 2):
    question = st.session_state.messages[idx]["content"]
    assistant_msg = st.session_state.messages[idx + 1] if idx + 1 < len(st.session_state.messages) else {"content": "Answer not found.", "source": "Unknown"}

    answer = assistant_msg.get("content", "Answer not found.")
    source = assistant_msg.get("source", "Unknown")

    pair_number = idx // 2 + 1
    expanded = (pair_number == total_pairs)  # latest expanded

    with st.expander(f"🔵 Question {pair_number}: {question}", expanded=expanded):
        st.markdown(f"**Answer:**\n\n{answer}")
        st.markdown(f"**Source:** {source}")
