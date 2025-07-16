# main.py
# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import numpy as np
import os

# --- 1. Model and Lifespan Management ---

# Global variable to hold the model, and path definition
model = None
MODEL_PATH = 'heart_disease_pipeline.pkl'
MODEL_PATH = os.path.abspath(MODEL_PATH)  # Ensure the path is absolute

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for the FastAPI app.
    This function is called once when the application starts up.
    It loads the ML model into memory and handles its cleanup on shutdown.
    """
    global model
    print("--- Application Startup: Loading Model ---")
    if not os.path.exists(MODEL_PATH):
        # In a real app, you might use more advanced logging
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        model = None
    else:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("--- Model Loaded Successfully ---")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    
    yield
    
    # --- Code below yield runs on shutdown ---
    print("--- Application Shutdown: Cleaning up ---")
    model = None


# --- 2. Application Initialization ---

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(
    title="Heart Disease Prediction API",
    description="An API to predict the presence of heart disease based on patient data.",
    version="1.0.0",
    lifespan=lifespan
)


# --- 3. Pydantic Model for Input Data ---

# Create a Pydantic model to define the structure and data types of the input
class PatientData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bp_s: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_angina: int
    oldpeak: float
    st_slope: int

    # Example data for the API documentation (Swagger UI)
    class Config:
        json_schema_extra = {
            "example": {
                "age": 52,
                "sex": 1,
                "chest_pain_type": 0,
                "resting_bp_s": 125,
                "cholesterol": 212,
                "fasting_blood_sugar": 0,
                "resting_ecg": 1,
                "max_heart_rate": 168,
                "exercise_angina": 0,
                "oldpeak": 1.0,
                "st_slope": 2
            }
        }

# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    """
    Root endpoint to welcome users and provide basic API information.
    """
    return {
        "message": "Welcome to the Heart Disease Prediction API",
        "docs_url": "/docs"
    }

@app.post('/predict')
def predict(data: PatientData):
    """
    Takes patient data as input and returns a heart disease prediction.
    """
    # This check ensures that the model was loaded successfully during startup.
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. It may have failed to load at startup."
        )

    try:
        # Convert the Pydantic model to a dictionary, then get the values
        # and convert them into a 2D numpy array for the model.
        input_data = np.array([list(data.model_dump().values())])

        # Make a prediction
        prediction_raw = model.predict(input_data)
        
        # It's good practice to convert numpy types to native Python types for JSON serialization
        prediction = int(prediction_raw[0])

        # Determine the human-readable message
        if prediction == 1:
            message = "Heart Disease Detected"
        else:
            message = "No Heart Disease Detected"

        # Return the prediction and the message
        return {
            "prediction_code": prediction,
            "prediction_message": message
        }
    except Exception as e:
        # Catch any other exceptions during prediction
        raise HTTPException(status_code=400, detail=f"An error occurred during prediction: {e}")

