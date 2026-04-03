from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import your existing agent
from retrieval.retrieval_engine import RetrievalEngine
from llm.provider import get_provider
from agent.pricing_agent import PricingAgent

engine = RetrievalEngine()
engine.load()
agent = PricingAgent(engine=engine, provider=get_provider())
agent._loaded = True


# Serve HTML
@app.get("/")
async def serve_ui():
    with open("pricing_ui.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/predict")
async def predict(
        description: str = Form(...),
        category_id: str = Form(""),
        image: UploadFile = File(...)
):
    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents))
        pil_image.save(tmp.name)
        image_path = tmp.name

    # Call your agent
    result = agent.recommend(
        description=description,
        image_url=image_path,
        category_id=category_id or None,
    )

    # Clean up
    os.unlink(image_path)

    # Format response
    return {
        "item_name": result.parsed_input.item_name,
        "condition": result.llm_result.condition_assessment,
        "evidence": result.llm_result.evidence,
        "recommended_price": result.strategies[0].price,
        "price_range": result.llm_result.price_range,
        "strategies": [
            {
                "emoji": s.emoji,
                "name": s.name,
                "price": s.price,
                "listing_price": s.listing_price,
                "days_to_sell": s.days_to_sell,
                "sale_probability": s.sale_probability,
                "explanation": s.explanation,
                "bargain_advice": s.bargain_advice,
            }
            for s in result.strategies
        ],
        "comparables": [
            {
                "title": c["title"],
                "price": f"₴{c['sold_price']:,.0f}" if c.get("sold_price") else f"₴{c['original_price']:,.0f}",
            }
            for c in result.comparables[:7]
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)