import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")