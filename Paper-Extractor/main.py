from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import webbrowser
import threading

from pdf_processing.process_pdf import process_pdf

app = FastAPI()


@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        temp_file_path = os.path.join(upload_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the PDF
        result = process_pdf(temp_file_path)

        # Delete uploaded file after processing
        os.remove(temp_file_path)

        return JSONResponse(content={"status": "success", "data": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Processing API!"}


# Automatically open Swagger Docs on startup
def open_swagger():
    webbrowser.open_new("http://127.0.0.1:8000/docs")


@app.on_event("startup")
def startup_event():
    threading.Timer(1.5, open_swagger).start()
