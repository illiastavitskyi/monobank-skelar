def build_vision_prompt(description: str) -> str:
    """Prompt for Gemini multimodal vision assessment.
    Returns JSON: {"coefficient": float (0.5–1.0), "reason": str (Ukrainian)}
    """
    return (
        f"You are an expert appraiser. Read the description: '{description[:500]}' "
        "and look at the item photo. Find any defects mentioned in the text "
        "(scratches, battery health, missing accessories, etc.) and combine them "
        "with what you see in the photo. "
        "Give a SINGLE unified condition coefficient: 0.5 = bad/broken, 1.0 = perfect. "
        "Respond with pure JSON only, no codeblocks. Reason MUST be in Ukrainian.\n"
        '{"coefficient": 0.85, "reason": "На фото видно знос, а в описі вказано про відсутність коробки."}'
    )