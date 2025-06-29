from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(transcript):
    try:
        prompt = (
            "Summarize the following meeting transcript in a clear and concise way:\n\n"
            f"{transcript}\n\nSummary:"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in summarization: {e}")
        # Fallback to a simple summary if OpenAI fails
        return f"Meeting transcript processed. Length: {len(transcript)} characters. Please check the full transcript for details."