from .schemas import ChatRequest, ChatResponse
from catbot.rag.basic_rag import BasicRAG

async def answer_question(
    input_payload: ChatRequest,
    rag: BasicRAG
) -> ChatResponse:
    """
    Wrapper function to handle a chat request and return a chat response."""
    response = rag.respond(input_payload.query)

    return ChatResponse(
        answer=response["response"],
        sources=response["sources"]
    )
