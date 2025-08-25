import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn

app = FastAPI()

# CORS setup
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     # "https://fascinating-rabanadas-929813.netlify.app",  # Netlify frontend
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fascinating-rabanadas-929813.netlify.app"],   # exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model pipeline
model_pipeline = joblib.load("carbon_pipeline.joblib")

class InputData(BaseModel):
    Body_Type: str
    Sex: str
    Diet: str
    How_Often_Shower: str
    Heating_Energy_Source: str
    Transport: str
    Vehicle_Type: str
    Social_Activity: str
    Frequency_of_Traveling_by_Air: str
    Waste_Bag_Size: str
    Energy_efficiency: str
    Monthly_Grocery_Bill: float
    Vehicle_Monthly_Distance_Km: float
    Waste_Bag_Weekly_Count: float
    How_Long_TV_PC_Daily_Hour: float
    How_Many_New_Clothes_Monthly: float
    How_Long_Internet_Daily_Hour: float
    Recycling: List[str] = []
    Cooking_With: List[str] = []

# ✅ Root route for Render testing
@app.get("/")
def root():
    return {"message": "Carbon Emission Predictor API is running."}

def preprocess_lists(df):
    for col in ["Recycling", "Cooking_With"]:
        df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else '')
        dummies = df[col].str.get_dummies(sep='|')
        dummies = dummies.add_prefix(col + "_")
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df

@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = {
            "Body Type": data.Body_Type,
            "Sex": data.Sex,
            "Diet": data.Diet,
            "How Often Shower": data.How_Often_Shower,
            "Heating Energy Source": data.Heating_Energy_Source,
            "Transport": data.Transport,
            "Vehicle Type": data.Vehicle_Type,
            "Social Activity": data.Social_Activity,
            "Frequency of Traveling by Air": data.Frequency_of_Traveling_by_Air,
            "Waste Bag Size": data.Waste_Bag_Size,
            "Energy efficiency": data.Energy_efficiency,
            "Monthly Grocery Bill": data.Monthly_Grocery_Bill,
            "Vehicle Monthly Distance Km": data.Vehicle_Monthly_Distance_Km,
            "Waste Bag Weekly Count": data.Waste_Bag_Weekly_Count,
            "How Long TV PC Daily Hour": data.How_Long_TV_PC_Daily_Hour,
            "How Many New Clothes Monthly": data.How_Many_New_Clothes_Monthly,
            "How Long Internet Daily Hour": data.How_Long_Internet_Daily_Hour,
            "Recycling": data.Recycling,
            "Cooking_With": data.Cooking_With
        }

        df = pd.DataFrame([input_dict])
        df = preprocess_lists(df)

        # Align with pipeline
        for col in model_pipeline.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model_pipeline.feature_names_in_]

        prediction = model_pipeline.predict(df)[0]
        return {"CarbonEmission": float(prediction)}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# uvicorn main:app --reload