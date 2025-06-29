from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def translate_text(text, target_language="English"):
    """Translates text to the target language using GPT-4."""
    if not text:
        return ""
    try:
        prompt = f"Translate the following text to {target_language}:\n\n{text}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": f"You are a translation expert. Translate the user's text to {target_language} accurately."},
                {"role": "user", "content": text}
            ],
            temperature=0,  # Low temperature for more deterministic translation
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return f"Error: Could not translate text."


def detect_and_translate_if_needed(query, target_language="English"):
    """
    Detects if a query is not in the target language, and if so, translates it.
    """
    try:
        prompt = (
            f"Is the following text in {target_language}? Answer with only 'yes' or 'no'.\n\n"
            f"Text: '{query}'"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3
        )
        is_target_language = response.choices[0].message.content.strip().lower()

        if 'no' in is_target_language:
            print(f"Query '{query}' is not in English. Translating...")
            return translate_text(query, target_language)

        print(f"Query '{query}' is already in English. No translation needed.")
        return query  # The query is already in the target language

    except Exception as e:
        print(f"Error in language detection, proceeding with original query: {e}")
        return query