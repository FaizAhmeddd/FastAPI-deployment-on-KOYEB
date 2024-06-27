from fastapi import FastAPI, APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from deepface import DeepFace
from pymongo import MongoClient, errors
from enum import Enum

model_name = "Dlib"

# Function to calculate embeddings using DeepFace
def get_dlib_embeddings(image_files):
    embeddings = []
    for image_file in image_files:
        try:
            # Convert upload image to numpy array using PIL
            image = Image.open(BytesIO(image_file.file.read()))

            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)

            # Calculate embeddings from numpy array
            embedding_dicts = DeepFace.represent(img_path=image_np, model_name=model_name, enforce_detection=False)
            for embedding_dict in embedding_dicts:
                if 'embedding' in embedding_dict:
                    embedding = embedding_dict['embedding']
                    embeddings.append(embedding)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {image_file.filename}: {e}")
    
    return np.mean(embeddings, axis=0)  # Return the mean of all embeddings


class employeeStatus(str, Enum):
    current_employee = "CurrentEmp"
    former_employee = "FormerEmp"
    on_leave = "OnLeave"
    inactive_employee = "InactiveEmp"

# Define Employee model
class Employee(BaseModel):
    emp_id: str
    first_name: str
    last_name: str
    email: str
    embedding: List[float]
    designation: Optional[str] = None
    status: employeeStatus

# Initialize FastAPI app and router
app = FastAPI()
router = APIRouter()

# Connect to MongoDB
client = MongoClient("mongodb+srv://askmeaboutcode:F4d42740@cluster0.n69qapt.mongodb.net/")
db = client["face_embeddings"]
employee_data = db["Employee"]

# Create a unique index on the emp_id field
employee_data.create_index("emp_id", unique=True)

# Endpoint to create employee with image uploads and embeddings
@router.post("/employees/")
async def create_employee_route(
    emp_id: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    images: List[UploadFile] = File(...),
    designation: Optional[str] = Form(None),
    status: employeeStatus = Form(...)
):
    try:
        # Generate embeddings for the images
        avg_embedding = get_dlib_embeddings(images)
        if avg_embedding.size == 0:
            raise HTTPException(status_code=400, detail="No embeddings were generated from the images.")
   
        employee = Employee(
            emp_id=emp_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            embedding=avg_embedding.tolist(),
            designation=designation,
            status=status
        )
        
        # Save the employee to the database
        data = employee.dict()
        result = employee_data.insert_one(data)
        if result.inserted_id:
            return JSONResponse(content={"id": emp_id, "message": "Employee created successfully"})
        else:
            raise HTTPException(status_code=500, detail="Failed to create employee")

    except errors.DuplicateKeyError:
        raise HTTPException(status_code=400, detail=f"Employee with emp_id {emp_id} already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create employee: {str(e)}")

# Include the router in the FastAPI app
app.include_router(router)

# Example ASGI server setup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
