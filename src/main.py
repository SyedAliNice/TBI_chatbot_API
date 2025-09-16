from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from chatbot_core import ClinicalChatbot

app = FastAPI(title="Clinical Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    try:
        chatbot = ClinicalChatbot()
        chatbot.model, chatbot.tokenizer, chatbot.pipeline, chatbot.llm = chatbot.load_model_and_tokenizer()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": chatbot is not None}

@app.post("/chat")
async def chat_endpoint(request: dict):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        session_id = request.get("session_id", "default")
        response = chatbot.process_message(request["message"], session_id)
        return {"response": response, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
