import os
from pydantic import BaseModel

class Settings(BaseModel):
    MODEL_ID: str = os.getenv("INTERNVL_MODEL_ID", "OpenGVLab/InternVL3_5-4B")
    DEVICE: str = os.getenv("INTERNVL_DEVICE", "cuda")
    DTYPE: str = os.getenv("INTERNVL_DTYPE", "float16")  # float16/bfloat16
    MAX_NEW_TOKENS: int = int(os.getenv("INTERNVL_MAX_NEW_TOKENS", "512"))
    PORT: int = int(os.getenv("MODEL_SERVER_PORT", "8000"))

settings = Settings()
