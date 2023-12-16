from fastapi import FastAPI
from model2000API import xgb_predictor

app = FastAPI()
class_dict = {0:'non-maintenance',1:'maintenance'}

# what is an emial related to
@app.post("/classfier")
def read_item(email: str):
    # print (id)
    return {'prediction': class_dict[xgb_predictor([email]).item()]}
    # return {"item_id": item_id}
