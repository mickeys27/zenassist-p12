from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI(title="ZenAssist API")

MODEL_PATH = "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class Claim(BaseModel):
    user_claim: str


@app.post("/tags")
async def predict_tag(claim: Claim):
    if not claim.user_claim.strip():
        raise HTTPException(status_code=400, detail="user_claim ne peut pas être vide")

    prediction = model.predict([claim.user_claim])[0]
    return {"tag": prediction}