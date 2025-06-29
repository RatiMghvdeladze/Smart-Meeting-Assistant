import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "meetings.json")


def load_data():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            return json.loads(content) if content else []
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_data(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_meeting(filename, transcript, transcript_segments, summary, action_items, decisions,
                 chunk_embeddings, visual_summary_url=None, speaker_stats=None, unique_speakers=0):
    """
    Save meeting with enhanced speaker information support

    Args:
        filename: Original filename
        transcript: Full transcript text
        transcript_segments: List of transcript segments with speaker info
        summary: AI-generated summary
        action_items: List of action items
        decisions: List of decisions
        chunk_embeddings: Embeddings for semantic search
        visual_summary_url: URL to visual summary (optional)
        speaker_stats: Dictionary of speaker statistics (NEW)
        unique_speakers: Number of unique speakers identified (NEW)
    """
    meetings = load_data()

    new_meeting = {
        "id": len(meetings) + 1 if meetings else 1,
        "filename": filename,
        "created_at": datetime.now().isoformat(),
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "title": f"Meeting {len(meetings) + 1 if meetings else 1}",
        "chunk_embeddings": chunk_embeddings,
        "visual_summary_url": visual_summary_url,
        # --- NEW SPEAKER FIELDS ---
        "speaker_stats": speaker_stats or {},
        "unique_speakers": unique_speakers
    }

    meetings.append(new_meeting)
    save_data(meetings)
    return new_meeting["id"]


def get_meetings():
    return load_data()


def get_meeting_by_id(meeting_id):
    meetings = load_data()
    for meeting in meetings:
        if meeting.get("id") == meeting_id:
            return meeting
    return None