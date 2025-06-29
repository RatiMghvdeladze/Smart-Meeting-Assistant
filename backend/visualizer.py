from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_visual_summary(summary_text):
    """Generates an image from a summary using DALL-E 3."""
    if not summary_text:
        return None
    try:
        prompt = (
            "Create a professional and minimalist infographic that visually represents the key points of a business meeting summary. "
            "Use abstract shapes, clean icons (like gears for processes, charts for data, checkmarks for decisions), and interconnected lines to show relationships. "
            "The style should be modern and corporate. Avoid using any text. The core themes from the meeting are: "
            f"'{summary_text[:600]}'" # Limit prompt length for API safety
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(f"Error creating visual summary with DALL-E 3: {e}")
        return None