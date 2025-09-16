import os

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "biomistral_tbi_finetuned_v2")
    MAX_SESSION_AGE_MINUTES = int(os.getenv("MAX_SESSION_AGE_MINUTES", 60))
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

settings = Settings()