from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
from app.model import load_model, predict
from app.preprocess import preprocess_image
from app.report import generate_report, create_pdf_report
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

app = FastAPI(title="CXR Inference Backend")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-vercel-app.vercel.app"],  # Update with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels (NIH Chest X-ray dataset)
CLASS_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Optimal thresholds for each disease
OPTIMAL_THRESHOLDS = {
    "0": 0.7080,  # Atelectasis
    "1": 0.7234,  # Cardiomegaly
    "2": 0.7239,  # Consolidation
    "3": 0.7236,  # Edema
    "4": 0.7092,  # Effusion
    "5": 0.7224,  # Emphysema
    "6": 0.7290,  # Fibrosis
    "7": 0.7307,  # Hernia
    "8": 0.6986,  # Infiltration
    "9": 0.7143,  # Mass
    "10": 0.5,    # Nodule
    "11": 0.7214, # Pleural_Thickening
    "12": 0.7251, # Pneumonia
    "13": 0.7295  # Pneumothorax
}

class ReportRequest(BaseModel):
    predictions: List[float]

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    predictions: List[float]
    inference_time: float
    probabilities: dict
    diagnoses: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and preprocess image
    image_bytes = await file.read()
    try:
        img_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
    
    # Run inference
    try:
        predictions, inference_time = predict(img_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    # Apply optimal thresholds to determine diagnoses
    diagnoses = {label: 1 if predictions[i] > OPTIMAL_THRESHOLDS[str(i)] else 0 for i, label in enumerate(CLASS_LABELS)}
    
    # Format response
    probabilities = {label: float(prob) for label, prob in zip(CLASS_LABELS, predictions)}
    return PredictionResponse(
        predictions=predictions.tolist(),
        inference_time=inference_time,
        probabilities=probabilities,
        diagnoses=diagnoses
    )

@app.post("/generate-report")
async def generate_report_endpoint(request: ReportRequest):
    predictions = request.predictions
    if len(predictions) != len(CLASS_LABELS):
        raise HTTPException(status_code=400, detail="Invalid predictions length")

    try:
        report_text = await generate_report(predictions, CLASS_LABELS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    report_id = str(uuid.uuid4())
    output_path = f"reports/report_{report_id}.pdf"
    os.makedirs("reports", exist_ok=True)
    
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, create_pdf_report, report_text, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    return FileResponse(output_path, media_type="application/pdf", filename=f"cxr_report_{report_id}.pdf")

@app.get("/health")
async def health():
    return {"status": "healthy"}