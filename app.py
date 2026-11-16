from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from model import analyze_csv, ensure_model

app = FastAPI(title="SentimentX")
ensure_model()  # loads the model at startup, training it first if needed


@app.get("/api/model")
def model_info():
    return ensure_model()["metrics"]


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    try:
        return analyze_csv(file.file)
    except ValueError as err:
        raise HTTPException(400, str(err))


# Serves the frontend at / (keep last so /api routes take priority).
app.mount("/", StaticFiles(directory="static", html=True), name="static")
