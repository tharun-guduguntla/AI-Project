import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import APIRouter, HTTPException, Query
from app.controllers.textsummarization_controller import ChatbotService
# from app.controllers.llmfactory import 
from app.controllers import CONFIG_FILE_PATH,set_config_location



# Load configuration
CONFIG_PATH = "config/llm.yaml"
config = set_config_location()
# config = load_config(CONFIG_PATH)

# PDF path
PDF_PATH = "data/Data_LLM.pdf"

# Initialize ChatbotService for the selected provider
PROVIDER_NAME = "gemini"  # Change to "openai" for OpenAI
chatbot_service = ChatbotService(pdf_path=PDF_PATH, config=config, provider_name=PROVIDER_NAME)

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
