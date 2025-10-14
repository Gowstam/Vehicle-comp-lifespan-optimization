from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_with_maintenance_ai(context: str, user_question: str) -> str:
    """
    Predictive Maintenance AI assistant that blends domain knowledge
    with telemetry context to provide both general and context-aware answers.
    """

    if not client:
        return " Chat feature disabled — missing API key."

    # --- Detect question type ---
    general_keywords = [
        "depend", "affect", "influence", "what causes", "how does",
        "role", "importance", "best practice", "parameter", "factor",
        "meaning", "impact", "relationship"
    ]
    is_general = any(re.search(k, user_question.lower()) for k in general_keywords)

    # --- Prompt Design ---
    if is_general:
        prompt = f"""
        You are an expert predictive maintenance engineer.
        The user has asked a general question about component behavior or dependencies.

        You have telemetry context from a real vehicle snapshot below.
        You should:
        - Use this data to illustrate patterns or correlations relevant to the question.
        - Provide a general explanation grounded in automotive maintenance and ML knowledge.
        - DO NOT just repeat or list all feature values.
        - Instead, synthesize insights: mention which telemetry variables are most relevant, and what their trends imply.

        Telemetry Context:
        {context}

        User Question:
        {user_question}

        Your response should be explanatory, combining general engineering understanding
        with observations from the provided telemetry.
        """
    else:
        # Fully contextual anomaly/failure reasoning
        prompt = f"""
        You are a predictive maintenance AI.
        Use the telemetry context below to explain the specific behavior, prediction, or anomaly requested.

        Focus on analyzing:
        - Which features show abnormal trends or deviations.
        - How those could lead to the observed or predicted failure.
        - Provide clear engineering reasoning, not just restating values.

        Telemetry Context:
        {context}

        User Question:
        {user_question}
        """

    # --- Generate response ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f" Chatbot error: {e}"
