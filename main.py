import os
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from langchain_openai import OpenAIEmbeddings

from rag import get_rag_chain
from feedback_classifier import tagging_chain

from dotenv import load_dotenv
load_dotenv()

ENDPOINT_QUERY_AI_MAX_SIZE = int(os.getenv("ENDPOINT_QUERY_AI_MAX_SIZE"))
ENDPOINT_EMBEDDING_TEXT_MAX_SIZE = int(os.getenv("ENDPOINT_EMBEDDING_TEXT_MAX_SIZE"))
ENDPOINT_EMBEDDING_MODEL = os.getenv("ENDPOINT_EMBEDDING_MODEL")
ENDPOINT_QUERY_FEEDBACK_CLASSIFY_MAX_SIZE = os.getenv("ENDPOINT_QUERY_FEEDBACK_CLASSIFY_MAX_SIZE")
app = FastAPI(
    title="Bora IA API",
    version="0.1.0"
)

@app.get("/feedback_classify")
async def feedback_classify(query: Annotated[str, Query(max_length=ENDPOINT_QUERY_FEEDBACK_CLASSIFY_MAX_SIZE)]):
    result = tagging_chain.invoke({"input": query})
    return result

@app.get("/ai")
async def ai(query: Annotated[str, Query(max_length=ENDPOINT_QUERY_AI_MAX_SIZE)]):
    response = get_rag_chain().invoke({"input": query})
    return response["answer"]

@app.post("/embedding")
async def embedding(text: Annotated[str, Query(max_length=ENDPOINT_EMBEDDING_TEXT_MAX_SIZE)]):
    embedder = OpenAIEmbeddings(model=ENDPOINT_EMBEDDING_MODEL)
    return embedder.embed_query(text)

@app.get("/ping")
async def root():
    return "Pong! :)"

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")