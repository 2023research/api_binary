from fastapi import FastAPI
from model2000API import xgb_predictor

app = FastAPI()

@app.get("/binaryclass/{email}")
async def binaryclassifier(email: str):
    return {'prediction':[email]}

@app.post("/binaryclassfier")
def read_item(email: str):
    # print (id)
    return {'prediction': xgb_predictor([email]).item()}
    # return {"item_id": item_id}
