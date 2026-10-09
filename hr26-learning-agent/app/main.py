from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.api.routes import router
from app.services.agnes_client import get_agnes_client


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    client = get_agnes_client()
    await client.close()


app = FastAPI(
    title="HR26 Learning Experiences Agent",
    description="Agentic AI system that turns scattered educational resources into adaptive, trustworthy learning experiences.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "name": "HR26 Learning Experiences Agent",
        "status": "ready",
        "docs": "/docs",
        "note": "Frontend not found. Place index.html in /frontend",
    }


# Serve any other static assets under /static if needed later
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
