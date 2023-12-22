from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model2000API import xgb_predictor
from email_location_126_onelabel_api_ready import cate_predictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class_dict = {0:'non-maintenance',1:'maintenance'}

# what is an emial related to
@app.post("/classfier")
def read_item(email: str):
    # print (id)
    is_maintenance = class_dict[xgb_predictor([email]).item()]
    if is_maintenance=='maintenance':
        cates = cate_predictor(email)
        issues = []
        for i in range(len(cates)):
            row = cates[i]
            row = row.split('@')
            issues.append({'area':row[0],'location':row[1],'Msubtype':row[2],'Mmaintype':row[3], 'summary':'this part will be generated by LLM in the near future'})
        
        return {'is_maintenance': is_maintenance, 'issues':issues,'summary':'this part will be generated by LLM in the future'}
    elif is_maintenance=='non-maintenance':
        return {'is_maintenance': is_maintenance, 'issues':[],'summary':'this part will be generated by LLM in the future'}
    # return {"item_id": item_id}
