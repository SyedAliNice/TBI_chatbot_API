from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
from typing import Dict

from . import models
from .chatbot_core import ClinicalChatbot

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Chatbot API",
    description="API for TBI clinical decision support chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot model on startup"""
    global chatbot
    try:
        chatbot = ClinicalChatbot()
        chatbot.model, chatbot.tokenizer, chatbot.pipeline, chatbot.llm = chatbot.load_model_and_tokenizer()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.get("/")
async def root():
    return {"message": "Clinical Chatbot API is running"}

@app.get("/health", response_model=models.HealthCheck)
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": chatbot is not None and chatbot.model is not None
    }

@app.post("/chat", response_model=models.ChatResponse)
async def chat_endpoint(request: models.ChatRequest):
    """Send a message to the chatbot and get a response"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use provided session_id or generate a new one
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process the message
        response = chatbot.process_message(request.message, session_id)
        
        # Clean up old sessions periodically
        if len(chatbot.sessions) > 100:  # Clean up if too many sessions
            chatbot.cleanup_old_sessions()
        
        return {
            "response": response,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific conversation session"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if session_id in chatbot.sessions:
        del chatbot.sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all conversation sessions"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    session_count = len(chatbot.sessions)
    chatbot.sessions.clear()
    return {"message": f"All {session_count} sessions cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)