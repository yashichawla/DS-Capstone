import os
from openai import OpenAI

SYSTEM_PROMPT = """
You are a skincare recommendation assistant.

Your job is to explain why a retrieved skincare product may fit a user's skin type and concern.

Rules:
- Use only the provided product information.
- Do not say "based on the provided information.
- Do not invent ingredients, benefits, claims, or medical advice.
- Do not mention any ingredient that is not listed.
- Use cautious wording such as "may help", "can support", or "is often used for".
- Keep the explanation to 2 concise sentences max.
- If the information is limited, say so briefly.
""".strip()


def get_groq_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def generate_product_explanation(
    *,
    skin_type: str,
    concern: str,
    product_name: str,
    product_type: str,
    step: str,
    price,
    matched_ingredients: list[str],
    excerpt: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    client = get_groq_client()

    matched_text = ", ".join(matched_ingredients[:12]) if matched_ingredients else "Not available"
    concern_text = concern.strip() if concern and concern.strip() else "Not specified"
    excerpt_text = excerpt[:500] if excerpt else "Not available"

    user_prompt = f"""
User profile:
- Skin type: {skin_type}
- Main concern: {concern_text}

Routine step: {step}

Product:
- Name: {product_name}
- Type: {product_type}
- Price: {price}
- Matched ingredients: {matched_text}
- Product excerpt: {excerpt_text}

Task:
Write a short explanation of why this product may suit the user's skin type and concern.
Focus on the product's role and the listed ingredients only.
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content
    return text.strip() if text else "This product was retrieved based on the available product information."