import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import APIRouter,FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.controllers.textsummarization_controller import ChatbotService
import os, sys

PDF_PATH = "data/Data_LLM.pdf"
chatbot_service = ChatbotService(pdf_path=PDF_PATH)

router = APIRouter()

@router.post("/assist-genie/api/v1/chat")
def chat_endpoint(
    question: str = Query(..., description="The user's question"),
    usecase_name: str = Query(..., description="The name of the use case"),
    usecase_key: str = Query(..., description="The key for the use case"),
) -> dict:
    """
    Process user question and return the response from the chatbot service.
    """
    try:
        response = chatbot_service.process_question(question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")