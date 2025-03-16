from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pickle as pkl
import joblib as lib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


api = FastAPI()
with open('PCOSPredictor.pkl', 'rb') as f:
    best_rf = pkl.load(f)
model=lib.load("PCOSPredictor.pkl")

# Define CORS middleware to allow cross-origin requests
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict it to specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class PCOs(BaseModel):
    Age:int
    Weight:int
    Height:int
    BloodGroup:int
    PeriodFrequency:int
    GainedWeight: int
    ExcessiveHair:int
    DarkSkin:int
    HairLoss:int
    FaceAcne:int
    FastFood:int
    RegularExercise:int
    MoodSwings:int
    RegularPeriods:int
    PeriodDuration:int
    BMI:float

@api.post("/predict")
async def predict(pcos:PCOs):
    features = np.array([[
        pcos.Age,
        pcos.Weight,
        pcos.Height,
        pcos.BloodGroup,
        pcos.PeriodFrequency,
        pcos.GainedWeight,
        pcos.ExcessiveHair,
        pcos.DarkSkin,
        pcos.HairLoss,
        pcos.FaceAcne,
        pcos.FastFood,
        pcos.RegularExercise,
        pcos.MoodSwings,
        pcos.RegularPeriods,
        pcos.PeriodDuration,
        pcos.BMI
    ]])
    prediction = model.predict_proba(features)[:,1]
    if prediction is None or len(prediction) == 0:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {prediction}")
    else:
        return {"pcos_risk_score": f"{prediction[0]:.2f}%"}

