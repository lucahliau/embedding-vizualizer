"""
Main application file for the embedding visualization service.
This file initializes and runs the FastAPI application.
"""

import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from src.api.routes import router
from src.config.settings import settings

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for reducing high-dimensional embeddings to 2D/3D for visualization",
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_PREFIX)

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Root endpoint - serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Run the application (when executed directly)
if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Server will run on http://{settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "src.main:app", 
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )