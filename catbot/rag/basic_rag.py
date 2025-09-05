"""File with the basic RAG implementation."""

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import Collection
import os

load_dotenv()  # Loads .env file into environment variables

API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT = """"
You are a knowledgeable assistant... and a cat. 
Use only the information provided in the context below to answer the user's question. 

If the answer cannot be found in the context, respond with: "Meow."
Do not use your own knowledge or make up any information.

Context:
{context}

Question: {query}
Answer:
"""


class BasicRAG:
    """A basic Retrieval-Augmented Generation (RAG) implementation using OpenAI's API and a ChromaDB collection."""

    def __init__(self, collection: Collection, prompt: str = PROMPT) -> dict:
        self.collection = collection
        self.prompt = prompt
        self.client = OpenAI(api_key=API_KEY)

    def send_request(self, prompt: str) -> str:
        """send a request to the OpenAI API and return the response"""
        completion = self.client.chat.completions.create(
            model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    def respond(self, query: str):
        """Respond to a user query using the RAG approach."""
        retrieved_documents = self.collection.query(query_texts=[query], n_results=5)

        filtered_documents = [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                retrieved_documents["documents"][0],
                retrieved_documents["metadatas"][0],
                retrieved_documents["distances"][0],
            )
            if dist < 0.4
        ]

        if filtered_documents:
            context_as_string = "\n\n".join(
                [doc["document"] for doc in filtered_documents]
            )
            response = self.send_request(
                prompt=self.prompt.format(context=context_as_string, query=query)
            )
            return {"response": response, "sources": filtered_documents}

        return {"response": "Hisssss", "sources": filtered_documents}
