from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ai_agent import get_response_from_ai_agent, get_available_providers, clear_conversation_history, handle_file_upload, get_conversation_files
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",   # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request schema
# -----------------------------
class ChatRequest(BaseModel):
    query: str
    system_prompt: str
    model_name: str
    model_provider: str
    allow_search: bool
    conversation_id: Optional[str] = None
    provider_config: Optional[Dict[str, Any]] = None

class ClearConversationRequest(BaseModel):
    conversation_id: str

class GetFilesRequest(BaseModel):
    conversation_id: str

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    response = get_response_from_ai_agent(
        query=request.query,
        system_prompt=request.system_prompt,
        model_name=request.model_name,
        model_provider=request.model_provider,
        allow_search=request.allow_search,
        conversation_id=request.conversation_id,
        provider_config=request.provider_config
    )
    return response

# File upload endpoint - ENHANCED VERSION
@app.post("/upload_file")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    conversation_id: str = Form(...)
):
    try:
        print(f"BACKEND DEBUG: Received file upload - filename: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}, type: {file.content_type}")
        
        # Read file content
        file_content = await file.read()
        print(f"BACKEND DEBUG: Read {len(file_content)} bytes from uploaded file")
        
        if len(file_content) == 0:
            return {
                "success": False,
                "error": "Empty file",
                "message": "The uploaded file appears to be empty (0 bytes)"
            }
        
        # Process the file
        result = handle_file_upload(
            file_content=file_content,
            filename=file.filename,
            conversation_id=conversation_id,
            file_type=file.content_type
        )
        
        print(f"BACKEND DEBUG: File processing result: {result}")
        return result
        
    except Exception as e:
        print(f"BACKEND DEBUG: Error in upload_file_endpoint: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error uploading file: {str(e)}"
        }

# Get uploaded files for conversation
@app.post("/get_files")
def get_files_endpoint(request: GetFilesRequest):
    try:
        files = get_conversation_files(request.conversation_id)
        print(f"BACKEND DEBUG: Retrieved {len(files)} files for conversation {request.conversation_id}")
        return {"files": files}
    except Exception as e:
        print(f"BACKEND DEBUG: Error in get_files_endpoint: {e}")
        return {"files": [], "error": str(e)}

# New endpoint to get available providers and models
@app.get("/providers")
def get_providers():
    return get_available_providers()

# Endpoint to clear conversation history
@app.post("/clear_conversation")
def clear_conversation_endpoint(request: ClearConversationRequest):
    return clear_conversation_history(request.conversation_id)

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Agent API is running"}

# Run with:
# uvicorn backend:app --reload --host 0.0.0.0 --port 9999