"""FastAPI service: predicts yearly developer salary (USD) and can be retrained with new data."""
import io
import threading
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from retrain import FEATURES, RetrainError, retrain

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

COUNTRIES = ["Argentina",
      "Australia",
      "Austria",
      "Bangladesh",
      "Belgium",
      "Brazil",
      "Bulgaria",
      "Canada",
      "China",
      "Colombia",
      "Czech Republic",
      "Denmark",
      "Egypt",
      "Finland",
      "France",
      "Germany",
      "Greece",
      "Hungary",
      "India",
      "Indonesia",
      "Iran, Islamic Republic of...",
      "Ireland",
      "Israel",
      "Italy",
      "Japan",
      "Kenya",
      "Lithuania",
      "Mexico",
      "Netherlands",
      "New Zealand",
      "Nigeria",
      "Norway",
      "Other",
      "Pakistan",
      "Poland",
      "Portugal",
      "Romania",
      "Russian Federation",
      "Slovakia",
      "South Africa",
      "Spain",
      "Sweden",
      "Switzerland",
      "Taiwan",
      "Turkey",
      "Ukraine",
      "United Kingdom of Great Britain and Northern Ireland",
      "United States of America",
      "Viet Nam",
      "Other"]            
ED_LEVELS = [
      "Associate degree (A.A., A.S., etc.)",
      "Bachelor\u2019s degree (B.A., B.S., B.Eng., etc.)",
      "Master\u2019s degree (M.A., M.S., M.Eng., MBA, etc.)",
      "Primary/elementary school",
      "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
      "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
      "Some college/university study without earning a degree",
      "Something else"]          # sorted unique EdLevel values
DEV_TYPES = [
      "Academic researcher",
      "Blockchain",
      "Cloud infrastructure engineer",
      "Data engineer",
      "Data or business analyst",
      "Data scientist or machine learning specialist",
      "Database administrator",
      "Designer",
      "DevOps specialist",
      "Developer Advocate",
      "Developer Experience",
      "Developer, AI",
      "Developer, QA or test",
      "Developer, back-end",
      "Developer, desktop or enterprise applications",
      "Developer, embedded applications or devices",
      "Developer, front-end",
      "Developer, full-stack",
      "Developer, game or graphics",
      "Developer, mobile",
      "Educator",
      "Engineer, site reliability",
      "Engineering manager",
      "Hardware Engineer",
      "Marketing or sales professional",
      "Other (please specify):",
      "Product manager",
      "Project manager",
      "Research & Development role",
      "Scientist",
      "Security professional",
      "Senior Executive (C-Suite, VP, etc.)",
      "Student",
      "System administrator"]          # sorted unique DevType values
REMOTE_WORK = [
      "Hybrid (some remote, some in-person)",
      "In-person",
      "Remote"]        # sorted unique RemoteWork values
ORG_SIZES = [
      "1,000 to 4,999 employees",
      "10 to 19 employees",
      "10,000 or more employees",
      "100 to 499 employees",
      "2 to 9 employees",
      "20 to 99 employees",
      "5,000 to 9,999 employees",
      "500 to 999 employees",
      "I don\u2019t know",
      "Just me - I am a freelancer, sole proprietor, etc."]          # sorted unique OrgSize values
YEARS_PRO_MAX = 15               # num_ranges from api_schema.json
YEARS_CODE_MAX = 34
NUM_LANGUAGES_MAX = 32


class SalaryInput(BaseModel):
    """One developer profile. Types are enforced and numeric fields have realistic ranges."""
    # strict: "5" (a string) or true is rejected for a number, and unknown extra fields are rejected
    model_config = ConfigDict(strict=True, extra="forbid")

    YearsCodePro: float = Field(..., ge=0, le=YEARS_PRO_MAX, description="Years coding professionally")
    YearsCode: float = Field(..., ge=0, le=YEARS_CODE_MAX, description="Total years coding")
    NumLanguages: int = Field(..., ge=0, le=NUM_LANGUAGES_MAX, description="Number of languages used")
    EdLevel: Literal[tuple(ED_LEVELS)]
    DevType: Literal[tuple(DEV_TYPES)]
    Country: Literal[tuple(COUNTRIES)]
    RemoteWork: Literal[tuple(REMOTE_WORK)]
    OrgSize: Literal[tuple(ORG_SIZES)]


class PredictionOut(BaseModel):
    predicted_salary_usd: float


app = FastAPI(title="Developer salary prediction API")

# CORS: only the browser origins listed here may call this API from a web page.
# Allowed: GET and POST (all the API uses), the Content-Type and Accept headers, and no cookies/credentials
# because the API has no login. Restricted: every other website, method and header. I did not use "*"
# because that would let any site on the internet call the API from a visitor's browser.
# A Flutter mobile app is not a browser, so CORS does not apply to it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://linear-regression-model-7hpc.onrender.com",  
        "http://localhost:8080",               
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)

_lock = threading.Lock()
model = joblib.load(MODEL_PATH)   # loaded once at startup, from the repository


@app.post("/predict", response_model=PredictionOut)
def predict(item: SalaryInput):
    row = pd.DataFrame([item.model_dump()])[FEATURES]
    with _lock:
        current = model
    pred_k = float(current.predict(row)[0])        # the model works in thousands of USD
    return PredictionOut(predicted_salary_usd=round(pred_k * 1000, 2))


@app.post("/retrain")
async def retrain_endpoint(file: UploadFile = File(...)):
    """Upload a CSV with the training columns (features + ConvertedCompYearly in USD); the model is updated."""
    global model
    try:
        new_df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as a CSV.")

    with _lock:
        current = model
    try:
        fresh, summary = retrain(current, new_df, COUNTRIES)   # logic lives in retrain.py
    except RetrainError as e:
        raise HTTPException(status_code=422, detail=str(e))

    joblib.dump(fresh, MODEL_PATH)
    with _lock:
        model = fresh                              # /predict now uses the updated model, no redeploy
    return summary