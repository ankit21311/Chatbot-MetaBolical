from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import final_result
import uvicorn

app = FastAPI(
    title="Medical Chatbot API",
    description="API endpoints for the Medical Chatbot using Llama-2",
    version="1.0.0"
)

class Query(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Medical Chatbot API is running"}

@app.post("/api/v1/chat")
async def chat(query: Query) -> ChatResponse:
    """
    Chat endpoint that accepts a question and returns an answer with sources
    """
    try:
        # Get response from the model
        response = final_result(query.question)
        
        # Extract answer and sources
        answer = response["result"]
        sources = response["source_documents"]
        
        return ChatResponse(
            answer=answer,
            sources=[{
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", "")
            } for doc in sources]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
