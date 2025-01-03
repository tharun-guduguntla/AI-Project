from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, sys

class ChatbotService:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def process_question(self, question: str) -> str:
        return f"{question}"

# Path to the PDF
PDF_PATH = "data/Data_LLM.pdf"
chatbot_service = ChatbotService(pdf_path=PDF_PATH)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatResponse(BaseModel):
    feedback: str

@app.get("/")
def read_root() -> dict:
    return {"message": "Welcome to the Assist Genie API!"}

@app.get("/favicon.ico")
def get_favicon() -> dict:
    return {"message": "No favicon available"}

@app.post("/assist-genie/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(
    question: str = Query(..., description="The user's question"),
    usecase_name: str = Query(..., description="The name of the use case"),
    usecase_key: str = Query(..., description="The key for the use case"),
) -> ChatResponse:
    """
    Endpoint to process user questions.
    """
    try:
        response = chatbot_service.process_question(question)
        return ChatResponse(feedback=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
