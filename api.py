from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.utils.model_setup import setup_model
from src.agents.manager_agent import create_manager_agent
from pydantic import BaseModel
import os
import uuid
from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Agent API",
    description="API for ML agent operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model = setup_model()
manager_agent = create_manager_agent(model)

class AgentRequest(BaseModel):
    prompt: str

@app.post("/model")
async def run_agent(request: AgentRequest):
    try:
        logger.info(f"Received request with prompt: {request.prompt}")
        result = manager_agent.run(request.prompt)
        logger.info(f"API Response: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Create a unique directory name using UUID
        upload_dir = str(uuid.uuid4())
        full_path = os.path.join("datasets", upload_dir)
        
        logger.info(f"Processing file upload to directory: {upload_dir}")
        
        # Create the directory if it doesn't exist
        os.makedirs(full_path, exist_ok=True)
        
        # Save each file
        for file in files:
            file_path = os.path.join(full_path, file.filename)
            logger.info(f"Saving file: {file.filename} to {file_path}")
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        response = {
            "status": "success",
            "directory": upload_dir,
            "message": f"Files uploaded successfully to {upload_dir}"
        }
        logger.info(f"Upload response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 