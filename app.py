"""Gradio UI — main entry point."""

import gradio as gr

from retrieval.retrieval_engine import RetrievalEngine
from llm.provider import get_provider
from agent.pricing_agent import PricingAgent

# Shared instances (loaded once at startup)
engine = RetrievalEngine()
agent = PricingAgent(engine=engine, provider=get_provider())


def get_category_choices() -> list[tuple[str, str]]:
    """Returns [(display_name, category_id), ...] for the dropdown."""
    return [("— Any category —", "")] + [
        (name, cid) for cid, name in sorted(engine.category_dict.items(), key=lambda x: x[1])
    ]


def run_pricing(image, description: str, category_id: str):
    import time
    import config

    if not description.strip():
        return "Please enter a product description.", "", "", "", ""
    if image is None:
        return "Please upload a product photo.", "", "", "", ""

    t_start = time.perf_counter()
    image_url = image if config.LLM_PROVIDER != "mock" else None

    result = agent.recommend(
        description=description,
        image_url=image_url,
        category_id=category_id or None,
    )

    # --- Context summary ---
    p = result.parsed_input
    parsed_md = f"**Item:** {p.item_name}"
    if p.quality:
        parsed_md += f"  |  **Condition:** {p.quality}"
    if p.quality_note:
        parsed_md += f"\n\n> ⚠️ {p.quality_note}"
    if p.additional_info:
        parsed_md += f"\n\n**Details:** {p.additional_info}"
    if p.assessed_price_range:
        parsed_md += f"\n\n**Pre-market estimate:** ₴{p.assessed_price_range[0]:,.0f} – ₴{p.assessed_price_range[1]:,.0f}"
    if result.category_name:
        parsed_md += f"\n\n**Category ({result.category_source}):** {result.category_name}"

    # --- Strategies ---
    strategies_md = ""
    for s in result.strategies:
        strategies_md += (
            f"### {s.emoji} {s.name}\n"
            f"**Sale price:** ₴{s.price:,.0f}  |  **List at:** ₴{s.listing_price:,.0f}\n\n"
            f"Expected time: {s.days_to_sell}  |  Sell probability: {round(s.sale_probability * 100)}%\n\n"
            f"_{s.explanation}_\n\n"
            f"> {s.bargain_advice}\n\n---\n"
        )

    price_range = result.llm_result.price_range
    range_md = f"**Price range:** ₴{price_range[0]:,.0f} – ₴{price_range[1]:,.0f}"

    evidence_md = "\n".join(f"- {e}" for e in result.llm_result.evidence)
    condition_md = f"**Condition:** {result.llm_result.condition_assessment}"

    comparables_md = ""
    if result.comparables:
        comparables_md = f"**{len(result.comparables)} similar listings found:**\n"
        for c in result.comparables:
            price = f"₴{c['sold_price']:,.0f} (sold)" if c.get("sold_price") else f"₴{c['original_price']:,.0f} ({c['status']})"
            comparables_md += f"- {c['title']} — {price}\n"

    elapsed = time.perf_counter() - t_start
    print(f"[timer] total processing: {elapsed:.2f}s")

    return parsed_md, strategies_md, range_md, f"{condition_md}\n\n{evidence_md}", comparables_md


def build_ui():
    category_choices = get_category_choices()

    with gr.Blocks(title="Monobazar Pricing Agent") as demo:
        gr.Markdown("# Monobazar AI Pricing Agent")
        gr.Markdown("Upload a photo and describe your item to get evidence-based price recommendations.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Product photo", type="filepath")
                description_input = gr.Textbox(label="Description", lines=4, placeholder="e.g. Nike Air Max 90, size 42, good condition, original box")
                category_input = gr.Dropdown(
                    label="Category (optional — leave blank to auto-detect)",
                    choices=get_category_choices(),
                    value="",
                )
                submit_btn = gr.Button("Get Price Recommendation", variant="primary")

            with gr.Column(scale=2):
                parsed_out = gr.Markdown(label="What we understood")
                strategies_out = gr.Markdown(label="Strategies")
                range_out = gr.Markdown(label="Price Range")

        with gr.Row():
            evidence_out = gr.Markdown(label="Evidence & Condition")
            comparables_out = gr.Markdown(label="Comparable Listings")

        submit_btn.click(
            fn=run_pricing,
            inputs=[image_input, description_input, category_input],
            outputs=[parsed_out, strategies_out, range_out, evidence_out, comparables_out],
        )

    return demo


if __name__ == "__main__":
    print("Loading data engine...")
    engine.load()
    agent._loaded = True  # noqa: SLF001
    print("Starting Gradio...")
    build_ui().launch(server_name="0.0.0.0", server_port=7860)