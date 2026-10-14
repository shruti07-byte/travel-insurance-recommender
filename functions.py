from groq import Groq
import os
import json


def validate_input(destination: str, trip_cost: int, duration: int, age: int, medical_condition: str) -> dict:
    """
    Validates traveler input before processing.
    Returns a dict with status 'ok' or 'error' and a list of error messages.
    """
    errors = []

    if not destination.strip():
        errors.append("Destination is required")

    if trip_cost <= 0:
        errors.append("Trip cost must be greater than 0")

    if duration <= 0:
        errors.append("Duration must be at least 1 day")

    if age <= 0 or age > 120:
        errors.append("Age must be between 1 and 120")

    if not medical_condition.strip():
        errors.append("Medical condition is required (use 'none' if not applicable)")

    return {
        "status": "ok" if not errors else "error",
        "errors": errors
    }


def get_coverages(destination: str, trip_cost: int, duration: int, age: int, medical_condition: str):
    """
    Determines relevant insurance coverages based on traveler profile.
    Returns a deduplicated list of coverage types and reasons.
    """
    coverages = []
    reasons = []

    destination = destination.lower().strip()
    medical = medical_condition.lower().strip()

    if medical not in ["none", "no", "nil", "n/a", ""]:
        coverages.append("Pre-existing Disease Coverage")
        coverages.append("Medical Expenses")
        reasons.append(f"traveler has a medical condition ({medical_condition})")

    if age >= 60:
        coverages.append("Senior Citizen Coverage")
        coverages.append("Medical Expenses")
        reasons.append(f"traveler is {age} years old (senior citizen)")

    expensive_keywords = [
        "usa", "united states", "america", "canada", "switzerland",
        "japan", "uk", "united kingdom", "germany", "france",
        "australia", "singapore", "dubai", "uae"
    ]
    if any(kw in destination for kw in expensive_keywords):
        coverages.append("Emergency Evacuation")
        coverages.append("Medical Expenses")
        reasons.append(f"{destination.title()} has high medical costs")

    if trip_cost >= 100000:
        coverages.append("Trip Cancellation")
        reasons.append(f"trip cost of ₹{trip_cost:,} needs financial protection")

    if duration >= 15:
        coverages.append("Travel Delay")
        reasons.append(f"long {duration}-day trip increases delay risk")

    adventure_keywords = [
        "nepal", "new zealand", "switzerland", "bali", "thailand",
        "vietnam", "himalayas", "ladakh", "manali"
    ]
    if any(kw in destination for kw in adventure_keywords):
        coverages.append("Adventure Sports Coverage")
        reasons.append(f"{destination.title()} is an adventure-prone destination")

    domestic_keywords = [
        "goa", "mumbai", "delhi", "bangalore", "bengaluru",
        "hyderabad", "chennai", "jaipur", "kolkata", "pune"
    ]
    if any(kw in destination for kw in domestic_keywords):
        coverages.append("Baggage Loss")
        reasons.append(f"domestic travel to {destination.title()}")

    if "Baggage Loss" not in coverages:
        coverages.append("Baggage Loss")

    coverages = list(dict.fromkeys(coverages))
    reasons = list(dict.fromkeys(reasons))

    return coverages, reasons


def build_prompt(destination: str, trip_cost: int, duration: int, age: int,
                 medical_condition: str, coverages: list, reasons: list) -> str:
    reasons_text = "; ".join(reasons)
    prompt = f"""You are a travel insurance advisor. Write a 2-sentence explanation for why these specific coverages were chosen for this traveler.

Traveler Profile:
- Destination: {destination}
- Age: {age} years
- Trip Cost: ₹{trip_cost:,}
- Trip Duration: {duration} days
- Medical Condition: {medical_condition}

Coverages Selected: {', '.join(coverages)}

Key reasons to mention: {reasons_text}

Rules:
- Be specific — mention the destination, age, trip cost, or medical condition directly
- Do NOT use generic phrases like "comprehensive protection" or "peace of mind"
- Do NOT list the coverages again
- Maximum 50 words
- Sound like a professional insurance advisor
"""
    return prompt


def get_llm_explanation(client: Groq, prompt: str) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": "You are a precise travel insurance advisor. Always give specific, data-driven explanations. Never be vague or generic."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def format_recommendation(coverages: list, explanation: str) -> str:
    coverage_text = "\n".join([f"• {c}" for c in coverages])
    return f"""
📋 Recommended Coverage:
{coverage_text}

💡 Why These Coverages:
{explanation}
"""


def get_recommendation(client: Groq, destination: str, trip_cost: int,
                        duration: int, age: int, medical_condition: str) -> str:
    coverages, reasons = get_coverages(destination, trip_cost, duration, age, medical_condition)
    prompt = build_prompt(destination, trip_cost, duration, age, medical_condition, coverages, reasons)
    explanation = get_llm_explanation(client, prompt)
    return format_recommendation(coverages, explanation)


def chatbot_reply(client: Groq, messages: list) -> dict:
    """
    Drives the chatbot conversation. Collects trip details step by step,
    then returns a recommendation when all fields are gathered.
    Returns: { "reply": str, "done": bool, "data": dict|None }
    """
    system_prompt = """You are Aria, a travel insurance advisor chatbot.

Collect these 5 details ONE AT A TIME:
1. destination
2. trip_cost (number in ₹)
3. duration (number of days)
4. age (number)
5. medical_condition (none, diabetes, heart condition, asthma, hypertension, cancer, other)

Rules:
- Ask ONE question per message — keep it very short (under 10 words)
- No long sentences, no explanations, no filler phrases
- Do NOT use any emoji in your messages
- Once you have ALL 5 details, respond with ONLY this JSON (no extra text):
{"destination":"USA","trip_cost":150000,"duration":20,"age":62,"medical_condition":"diabetes"}
- Start by greeting briefly and asking for destination"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.5,
        max_tokens=200
    )

    reply = response.choices[0].message.content.strip()

    # Check if reply is a JSON with all required fields
    try:
        # Strip markdown code blocks if any
        clean = reply.strip().strip("```json").strip("```").strip()
        data = json.loads(clean)
        required = {"destination", "trip_cost", "duration", "age", "medical_condition"}
        if required.issubset(data.keys()):
            return {"reply": reply, "done": True, "data": data}
    except Exception:
        pass

    return {"reply": reply, "done": False, "data": None}