import os
import tempfile
import traceback
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse

from app.agent import PricingAgent
from config import FRONTEND_HTML, HOST, PORT, RELOAD

app = FastAPI()
agent = PricingAgent()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FRONTEND_HTML.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze_item(title: str = Form(...), files: List[UploadFile] = File(...)):
    tmp_paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await file.read())
                tmp_paths.append(tmp.name)

        result = agent.analyze(description=title, image_paths=tmp_paths)
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD)