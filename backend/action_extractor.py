from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_action_items(transcript):
    """
    Extracts action items and decisions using GPT-4 with formal Function Calling (tools).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "log_meeting_outcomes",
                "description": "Logs the action items and key decisions from a meeting transcript.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_items": {
                            "type": "array",
                            "description": "A list of action items to be completed.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string", "description": "The specific task to be done."},
                                    "assignee": {"type": "string",
                                                 "description": "The person or team responsible. Default to 'Unassigned' if not mentioned."},
                                    "due_date": {"type": "string",
                                                 "description": "The due date, e.g., 'EOD Friday'. Default to 'None' if not mentioned."}
                                },
                                "required": ["task", "assignee"]
                            }
                        },
                        "decisions": {
                            "type": "array",
                            "description": "A list of key decisions made during the meeting.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "decision_summary": {"type": "string",
                                                         "description": "A concise summary of the decision made."},
                                    "owner": {"type": "string",
                                              "description": "The person who owns or proposed the decision. Default to 'Group Decision' if not specified."}
                                },
                                "required": ["decision_summary"]
                            }
                        }
                    },
                    "required": ["action_items", "decisions"],
                },
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are an expert meeting assistant. Analyze the transcript and extract all action items and key decisions using the provided tools. If no action items or decisions are found, call the function with empty arrays."},
                {"role": "user", "content": transcript}
            ],
            tools=tools,
            tool_choice="auto"  # Let the model decide when to call the function
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # We only expect one tool call for this setup
            function_args = json.loads(tool_calls[0].function.arguments)
            return function_args  # Returns a dict like {'action_items': [...], 'decisions': [...]}

        # If the model decides there's nothing to extract, return an empty structure
        print("Warning: GPT-4 did not call the function. Returning empty outcomes.")
        return {"action_items": [], "decisions": []}

    except Exception as e:
        print(f"Error in action item/decision extraction: {e}")
        return {"action_items": [], "decisions": []}