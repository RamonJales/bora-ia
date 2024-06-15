from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="Bora IA API",
    version="0.1.0"
)

@app.get("/ping")
async def root():
    return "Pong! :)"

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")