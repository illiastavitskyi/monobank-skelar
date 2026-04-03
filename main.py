import tempfile
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

from agent_inference import PricingAgent

load_dotenv()

app = FastAPI()
agent = PricingAgent()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
async def analyze_item(
    title: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        tmp_image_paths = []
        try:
            # Save uploaded images temporarily
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_image_paths.append(tmp_file.name)

            # We call the new method analyze_for_frontend
            result = agent.analyze_for_frontend(description=title, image_paths=tmp_image_paths)
            return JSONResponse(content=result)
        finally:
            for p in tmp_image_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
