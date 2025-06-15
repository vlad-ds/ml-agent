from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.utils.model_setup import setup_model
from src.agents.analysis_agent import create_analysis_agent
from src.agents.modeling_agent import create_modeling_agent
from src.agents.manager_agent import create_manager_agent
from pydantic import BaseModel
import os
import uuid
from typing import List

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
analysis_agent = create_analysis_agent(model)
modeling_agent = create_modeling_agent(model)
manager_agent = create_manager_agent(model, [analysis_agent, modeling_agent])

class AgentRequest(BaseModel):
    prompt: str

@app.post("/model")
async def run_agent(request: AgentRequest):
    try:
        result = manager_agent.run(request.prompt)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Create a unique directory name using UUID
        upload_dir = str(uuid.uuid4())
        full_path = os.path.join("datasets", upload_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(full_path, exist_ok=True)
        
        # Save each file
        for file in files:
            file_path = os.path.join(full_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        return {
            "status": "success",
            "directory": upload_dir,
            "message": f"Files uploaded successfully to {upload_dir}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 