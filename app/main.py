from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.chatbot_service import ChatbotService
import os,sys

"""
reffering to data folder for reading the data engineering role based document read
"""
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))


"""
reading data from the data folder
"""
PDF_PATH = "data/Data_LLM.pdf"
chatbot_service = ChatbotService(pdf_path=PDF_PATH)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatResponse(BaseModel):
    feedback: str

@app.get("/")
def read_root()->dict:
    return {"message": "Welcome to the Assist Genie API!"}

@app.get("/favicon.ico")
def get_favicon()->dict:
    return {"message": "No favicon available"}

@app.post("/assist-genie/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(
    question: str = Query(..., description="The user's question"),
    usecase_name: str = Query(..., description="The name of the use case"),
    usecase_key: str = Query(..., description="The key for the use case"),
    authorization: str = Header(..., description="Authorization token")
) ->None:
    "Step to authenticate the user for the access"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        # Here you can use the `usecase_name` and `usecase_key` as needed
        response = chatbot_service.process_question(question)
        return ChatResponse(feedback=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
