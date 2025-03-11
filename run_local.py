"""
Script to run the embedding visualization service locally.
This is useful for development and testing.
"""

import uvicorn
import logging
import os
import webbrowser
from threading import Timer

from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# For local development, use localhost
HOST = "127.0.0.1"  
PORT = settings.PORT
APP_MODULE = "src.main:app"  # Path to the FastAPI app

def open_browser():
    """Open the browser after a short delay to give the server time to start."""
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} in development mode")
    logger.info(f"Server will run on http://{HOST}:{PORT}")
    
    # Open browser after a short delay
    Timer(1.5, open_browser).start()
    
    # Run the application with local settings
    uvicorn.run(
        APP_MODULE, 
        host=HOST, 
        port=PORT, 
        reload=True  # Enable auto-reload for development
    )