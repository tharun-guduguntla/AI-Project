from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.chatbot_service import ChatbotService

# Initialize chatbot service
PDF_PATH = "Data_LLM.pdf"  # Replace with your PDF file path
chatbot_service = ChatbotService(pdf_path=PDF_PATH)

# FastAPI app
app = FastAPI()

# Serve static files (e.g., favicon.ico)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request and response
class Usecase(BaseModel):
    name: str
    key: str

class ChatRequest(BaseModel):
    question: str
    usecase: Usecase

class ChatResponse(BaseModel):
    feedback: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Assist Genie API!"}

@app.get("/favicon.ico")
def get_favicon():
    return {"message": "No favicon available"}

@app.post("/assist-genie/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, authorization: str = Header(...)):
    # Authorization check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Process the question
    try:
        response = chatbot_service.process_question(request.question)
        return ChatResponse(feedback=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
