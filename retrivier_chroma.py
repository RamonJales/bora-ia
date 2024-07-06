from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from persist_chroma import load_chroma

load_dotenv()

db = load_chroma()

retriever = db.as_retriever()