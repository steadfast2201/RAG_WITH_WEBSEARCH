import base64
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_web_processing import add_to_db
from web_search import decode_filename_to_url, remove_temp_files
from config_settings import CHUNK_OVERLAP, CHUNK_SIZE

# The code loads local documents (from a folder), chunks them into manageable text pieces, indexes them into a vector database using embeddings, 
# performs similarity search for a query, and generates a contextualized prompt for an LLM to answer the query.

                                    # Local TXT files → Load → Chunk → Embed → Vector DB
                                    #                         ↓
                                    #                       Query
                                    #                         ↓
                                    #             Similarity Search on Chunks
                                    #                         ↓
                                    #            Generate Context-Aware Prompt
                                    #                         ↓
                                    #       [Prompt, List of Source URLs] returned



def load_documents(download_dir: str = "./downloaded") -> list[Document]:
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        download_dir,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
    )
    return loader.load()


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)


def generate_prompt(query: str, embedding_function):
    documents = load_documents()
    chunks = split_documents(documents)
    db = add_to_db(chunks, embedding_function)
    results = db.similarity_search_with_score(query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an intelligent assistant designed to answer user questions accurately and using the following context document.
        Context:
        {context}

        ---

        Now answer the following question: 
        {question}
        """
    )

    sources = [
        decode_filename_to_url(doc.metadata.get("source", "Unknown"))[11:]
        for doc, _score in results
    ]
    prompt = prompt_template.format(context=context_text, question=query)

    remove_temp_files()
    return prompt, str(sources)
