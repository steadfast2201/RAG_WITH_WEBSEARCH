import time
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL_NAME
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document

def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME):
    return OllamaEmbeddings(model=model_name)

def add_to_db(chunks, embedding_function) -> InMemoryVectorStore:
    db = InMemoryVectorStore(embedding=embedding_function)
    db.add_documents(chunks)
    return db

def search_in_file(queries, file_path, embedding_function: Embeddings, timeout=30):
    """
    Search relevant information in uploaded file using LangChain.
    If something is found within the timeout period, return (prompt, sources).
    Otherwise, return None.
    """
    # 1. Load file
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(docs)

    # 3. Embed and build vector store
    vectorstore = FAISS.from_documents(split_docs, embedding_function)

    # 4. Search for each query with a timeout
    start_time = time.time()
    for query in queries:
        if time.time() - start_time > timeout:
            break

        results: list[Document] = vectorstore.similarity_search(query, k=4)

        if results:
            content = "\n\n".join([doc.page_content for doc in results])
            sources = "\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}" for doc in results])
            prompt = f"Use the following information to answer the question:\n\n{content}\n\nQuestion: {query}"
            return prompt, sources

    return None

