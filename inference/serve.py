from fastapi import FastAPI, Request
import uvicorn, os, json
from inference.predictor import Predictor

app = FastAPI()
predictor = Predictor()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(req: Request):
    body = await req.json()
    text = body.get("text", "")
    return {"entities": predictor.predict(text)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
