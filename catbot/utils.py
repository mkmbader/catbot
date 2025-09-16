"""This contains utility functions for the catbot project."""
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from dotenv import load_dotenv
import os

load_dotenv() 
API_KEY = os.getenv("OPENAI_API_KEY")

def get_evaluation_models(model_name: str = "gpt-4.1-mini", embedding_model_name: str = "text-embedding-3-small"):
    """Returns the LLM and embeddings model for evaluation."""

    # ragas llm and embeddings utilities
    llm = ChatOpenAI(
        model= model_name,
        openai_api_key=API_KEY,
    )

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=API_KEY,
    )

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)