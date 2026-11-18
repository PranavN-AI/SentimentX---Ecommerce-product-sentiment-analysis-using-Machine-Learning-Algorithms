import re
from datetime import datetime

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from exporter import build_workbook
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


@app.post("/api/export/xlsx")
async def export_xlsx(payload: dict = Body(...)):
    try:
        content = build_workbook(payload)
    except (KeyError, TypeError, ValueError) as err:
        raise HTTPException(400, f"Could not build the workbook: {err}")
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", str(payload.get("file_name") or "analysis").rsplit(".", 1)[0]).strip("-")
    name = f"sentimentx-{base or 'analysis'}-{datetime.now():%Y%m%d}.xlsx"
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# Serves the frontend at / (keep last so /api routes take priority).
app.mount("/", StaticFiles(directory="static", html=True), name="static")
