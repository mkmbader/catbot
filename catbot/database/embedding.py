"""Create embeddings for documents and store them in a vector database."""

import chromadb
from langchain.schema import Document


def create_data_base(
    sections: list[Document],
    embeddings_function,
    db_file_path: str = "catbot/database/chroma",
):
    """Creates a ChromaDB database and stores the documents with their embeddings."""

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
