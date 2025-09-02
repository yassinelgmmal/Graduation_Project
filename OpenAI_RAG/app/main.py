from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.routers import qa, summary
import uvicorn
import os
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("🚀 Starting Multimodal RAG API...")
    logger.info("📁 Ensuring required directories exist...")
    
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    logger.info("✅ Application startup complete")
    
    try:
        yield
    except Exception as e:
        logger.error(f"Error during application lifecycle: {e}")
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Multimodal RAG API...")
        logger.info("✅ Shutdown complete")

app = FastAPI(
    title="Multimodal RAG API",
    description="API for ingesting scientific papers and querying them via retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(qa.router, prefix="/api/v1")
app.include_router(summary.router, prefix="/api/v1")

# Root path
@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal RAG system.", "version": "0.1.0"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8010,
            reload=True,
            log_level="info",
            loop="asyncio"  # Explicitly use asyncio event loop
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except asyncio.CancelledError:
        logger.info("🛑 Server tasks cancelled during shutdown")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        raise