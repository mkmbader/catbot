"""Create embeddings for documents and store them in a vector database."""

import chromadb
from langchain.schema import Document
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file into environment variables

API_KEY = os.getenv("OPENAI_API_KEY")


def embedding_function(model_name: str) -> embedding_functions.OpenAIEmbeddingFunction:
    """Creates an embedding function using OpenAI's API."""

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY, model_name=model_name
    )


def create_data_base(
    sections: list[Document],
    model_name="text-embedding-3-small",
    db_file_path: str = "catbot/database/chroma",
):
    """Creates a ChromaDB database and stores the documents with their embeddings."""

    embeddings_function = embedding_function(model_name=model_name)

    chroma_client = chromadb.PersistentClient(path=db_file_path)

    collection = chroma_client.create_collection(
        name="its_all_about_cats",
        embedding_function=embeddings_function,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=[doc.page_content for doc in sections],
        metadatas=[doc.metadata for doc in sections],
        ids=[doc.id for doc in sections],
    )
