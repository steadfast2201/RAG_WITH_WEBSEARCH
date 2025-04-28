import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from config_settings import EMBEDDING_MODEL_NAME
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


def calculate_similarity(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def search_in_file(queries, file_path, embedding_function: Embeddings, timeout=30):
    """
    Search relevant information in uploaded file using LangChain.
    If something is found within the timeout period, return (prompt, sources, similarity_score).
    Otherwise, return None.
    """
    # 1. Load file
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 3. Embed and build vector store
    vectorstore = FAISS.from_documents(split_docs, embedding_function)

    # 4. Get combined query embedding
    full_query = " ".join(queries)
    query_embedding = embedding_function.embed_query(full_query)

    # 5. Search and score
    start_time = time.time()
    best_result = None
    best_score = -1

    for query in queries:
        if time.time() - start_time > timeout:
            break

        results: list[Document] = vectorstore.similarity_search(query, k=4)
        for doc in results:
            doc_embedding = embedding_function.embed_query(doc.page_content)
            score = calculate_similarity(query_embedding, doc_embedding)

            if score > best_score:
                best_score = score
                best_result = doc

    # 6. Threshold check to ensure relevance
    if best_result and best_score > 0.4:
        content = best_result.page_content
        source = best_result.metadata.get("source", "Unknown")
        prompt = f"Use the following information to answer the question:\n\n{content}\n\nQuestion: {full_query}"
        sources = f"Source: {source}"
        return prompt, sources, best_score
    else:
        return None
