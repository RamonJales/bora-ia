from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from rag import rag_chain

app = FastAPI(
    title="Bora IA API",
    version="0.1.0"
)

@app.get("/ai")
async def ai(query: str):
    response = rag_chain.invoke({"input": query})
    return response["answer"]

@app.get("/ping")
async def root():
    return "Pong! :)"

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")