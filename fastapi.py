import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

with open('final_chain.pkl', 'rb') as f:
    final_chain = pickle.load(f)

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    sentiment: str
    automated_email: str

@app.post("/generate_email", response_model=ReviewResponse)
async def generate_email(request: ReviewRequest):
    try:
        result = final_chain.invoke(request.review)
        response = ReviewResponse(
            review=request.review,
            sentiment=result['sentiment'],
            automated_email=result['automated_email']
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
