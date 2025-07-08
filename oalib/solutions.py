"""Library with OpenAI API solutions as functions

References:
For building code: https://platform.openai.com/docs/guides/chat
"""

import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, OpenAIError

# Initialize the OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI()

def submit_question(text: str) -> str:
    """Submit a question to the OpenAI API, trying GPT-4 first, then falling back to GPT-3.5-turbo."""
    models_to_try = ["gpt-4", "gpt-3.5-turbo"]
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            # If model not found, try next model
            if "model_not_found" in str(e):
                continue
            # For other errors, raise immediately
            raise e
    raise RuntimeError("No suitable model found or accessible.")

def create_code(text: str, language: str) -> str:
    """Create code in the specified language, with fallback between GPT-4 and GPT-3.5-turbo."""
    system_msg = f"You are a helpful assistant that writes clean, efficient {language} code."
    user_msg = f"Please write {language} code for the following task:\n{text}"

    models_to_try = ["gpt-4", "gpt-3.5-turbo"]
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            if "model_not_found" in str(e):
                continue
            raise e
    raise RuntimeError("No suitable model found or accessible.")