from fastapi import FastAPI, Depends, HTTPException, status
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from catbot.database.embedding import embedding_function
from catbot.rag.basic_rag import BasicRAG
from catbot.database.downloader import download_and_chunk_wikipedia_articles
from catbot.database.embedding import create_data_base
from chromadb import PersistentClient
from api.schemas import ChatRequest, ChatResponse
from api.chat_wrapper import answer_question



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager for the application lifespan.
    Initializes the RAG chatbot on startup and cleans it up on shutdown.
    """
    print("FastAPI application starting up (lifespan manager)...")

    articles = download_and_chunk_wikipedia_articles(['Cat'])

    try:
        create_data_base(articles)
        print("Database created successfully.")
    except:
        print("Database already exists, loaded existing database.")

    client = PersistentClient(path='catbot/database/chroma')
    collection = client.get_collection("its_all_about_cats", 
                                       embedding_function=embedding_function(model_name="text-embedding-3-small"))

    rag = BasicRAG(
        collection=collection
    )

    
    app.state.rag = rag
    print("RAG Chatbot instance created and stored in app.state.")

    yield 

    print("FastAPI application shutting down (lifespan manager)...")

    if app.state.rag:
        await app.state.rag.close()
    print("RAG Chatbot instance cleaned up.")
    print("FastAPI application shut down gracefully.")


app = FastAPI(
    title="RAG Chatbot API",
    description="API for a Retrieval-Augmented Generation chatbot, integrating a custom RAG class.",
    version="0.1.0",
    lifespan=lifespan 
)


# --- 4. Dependency for RAG Chatbot Instance ---
# This function now retrieves the chatbot instance from app.state.

async def get_rag_chatbot() -> BasicRAG:
    """
    Provides the RAG chatbot instance to API endpoints from app.state.
    """
    # This assumes `app` is available in the global scope where this dependency is defined
    # FastAPI's Depends system handles this for you.
    # In a real setup, `Request` object can be used to access `app.state`
    # from fastapi import Request
    # return request.app.state.rag

    # For simplicity, if `app` is globally defined and accessible:
    if not hasattr(app.state, "rag") or app.state.rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Chatbot is not initialized yet. Please try again in a moment."
        )
    return app.state.rag


@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(
    request_body: ChatRequest, 
    rag: BasicRAG = Depends(get_rag_chatbot)
):
    """
    Send a natural language query to the RAG chatbot and get a generated answer.
    """
    input_payload = ChatRequest.parse_obj(request_body)
    print(f"Received chat request: {input_payload}")
    try:
        # answer = await rag.respond(input_payload.query)
        response_object = await answer_question(
            input_payload=input_payload,
            rag=rag
        )

        return response_object
    except Exception as e:
        print(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred while processing your query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint to verify the API is running and the chatbot is initialized.
    """
    chatbot_instance = getattr(app.state, "rag", None)
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot_instance is not None,
        # "llm_model": chatbot_instance.llm_model if chatbot_instance else "N/A"
    }

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot API! Visit /docs for interactive documentation."}