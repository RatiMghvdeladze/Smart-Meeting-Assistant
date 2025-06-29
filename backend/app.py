from flask import Flask, request, jsonify, render_template
import os
import threading
import uuid
from .whisper_transcriber import transcribe_with_speakers, get_speaker_summary
from .summarizer import summarize_text
from .action_extractor import extract_action_items
from .visualizer import create_visual_summary
from .semantic_searcher import create_embeddings_for_meeting, search_meetings, find_similar_meetings
from .database import save_meeting, get_meetings, get_meeting_by_id
from .translator import detect_and_translate_if_needed

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processing_status = {}


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except:
        return "Could not find index.html", 404


def process_meeting_async(file_path, filename, task_id, speaker_settings=None):
    """
    Enhanced processing function with speaker identification

    Args:
        file_path: Path to audio file
        filename: Original filename
        task_id: Unique task identifier
        speaker_settings: Dict with min_speakers, max_speakers (optional)
    """
    try:
        processing_status[task_id] = {'status': 'processing', 'step': 'transcribing'}

        # --- UPDATED: Use WhisperX with speaker identification ---
        min_speakers = speaker_settings.get('min_speakers') if speaker_settings else None
        max_speakers = speaker_settings.get('max_speakers') if speaker_settings else None

        transcript_result = transcribe_with_speakers(file_path, min_speakers, max_speakers)

        # Extract data from WhisperX result
        transcript = transcript_result["text"]
        transcript_segments = transcript_result["segments"]
        speaker_stats = transcript_result["speaker_stats"]
        unique_speakers = transcript_result["unique_speakers"]

        print(f"Transcription completed: {unique_speakers} speakers identified")

        processing_status[task_id].update({
            'step': 'summarizing',
            'speaker_count': unique_speakers,
            'speaker_stats': speaker_stats
        })

        # --- ENHANCED: Include speaker information in summary ---
        speaker_summary = get_speaker_summary(speaker_stats)
        summary_prompt = f"MEETING PARTICIPANTS:\n{speaker_summary}\n\nTRANSCRIPT:\n{transcript}"
        summary = summarize_text(summary_prompt)

        processing_status[task_id].update({'step': 'embedding'})
        searchable_text = f"Summary: {summary}\n\nSpeaker Info: {speaker_summary}\n\nTranscript: {transcript}"
        chunk_embeddings = create_embeddings_for_meeting(searchable_text)

        processing_status[task_id].update({'step': 'visualizing'})
        visual_summary_url = create_visual_summary(summary)

        processing_status[task_id].update({'step': 'extracting_actions'})
        outcomes = extract_action_items(transcript)
        action_items, decisions = outcomes.get('action_items', []), outcomes.get('decisions', [])

        processing_status[task_id].update({'step': 'saving'})

        # --- UPDATED: Save meeting with speaker information ---
        meeting_id = save_meeting(
            filename, transcript, transcript_segments, summary,
            action_items, decisions, chunk_embeddings, visual_summary_url,
            speaker_stats=speaker_stats, unique_speakers=unique_speakers
        )

        processing_status[task_id] = {
            'status': 'completed',
            'step': 'done',
            'meeting_id': meeting_id,
            'transcript_segments': transcript_segments,
            'summary': summary,
            'action_items': action_items,
            'decisions': decisions,
            'visual_summary_url': visual_summary_url,
            'speaker_stats': speaker_stats,
            'unique_speakers': unique_speakers
        }

    except Exception as e:
        print(f"Error in process_meeting_async: {str(e)}")
        processing_status[task_id] = {'status': 'error', 'message': str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/upload', methods=['POST'])
def upload_meeting():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # --- NEW: Get speaker settings from request ---
    speaker_settings = {}
    if request.form.get('min_speakers'):
        try:
            speaker_settings['min_speakers'] = int(request.form.get('min_speakers'))
        except ValueError:
            pass

    if request.form.get('max_speakers'):
        try:
            speaker_settings['max_speakers'] = int(request.form.get('max_speakers'))
        except ValueError:
            pass

    task_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=process_meeting_async,
        args=(file_path, filename, task_id, speaker_settings)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'task_id': task_id})


@app.route('/status/<task_id>')
def get_status(task_id):
    return jsonify(processing_status.get(task_id, {'status': 'not_found'}))


@app.route('/meetings')
def list_meetings():
    meetings = get_meetings()
    return jsonify([{
        "id": m["id"],
        "title": m["title"],
        "created_at": m["created_at"],
        "filename": m["filename"],
        "summary": m.get("summary", "")[:150],
        "unique_speakers": m.get("unique_speakers", 0),
        "has_speaker_info": bool(m.get("speaker_stats"))
    } for m in meetings])


@app.route('/meetings/<int:meeting_id>')
def get_single_meeting(meeting_id):
    meeting = get_meeting_by_id(meeting_id)
    if not meeting:
        return jsonify({'error': 'Meeting not found'}), 404

    # --- ENHANCED: Include speaker summary in response ---
    if meeting.get('speaker_stats'):
        meeting['speaker_summary'] = get_speaker_summary(meeting['speaker_stats'])

    return jsonify(meeting)


@app.route('/meetings/<int:meeting_id>/similar')
def get_similar_meetings(meeting_id):
    all_meetings = get_meetings()
    recommendations = find_similar_meetings(meeting_id, all_meetings)
    return jsonify(recommendations)


@app.route('/search', methods=['POST'])
def search_route():
    data = request.get_json()
    original_query = data.get('query')
    if not original_query:
        return jsonify({'results': []})

    english_query = detect_and_translate_if_needed(original_query, "English")
    results = search_meetings(english_query, get_meetings())

    return jsonify({
        'results': results,
        'translated_query': english_query if original_query.lower() != english_query.lower() else None
    })


# --- NEW: Endpoint for speaker statistics ---
@app.route('/meetings/<int:meeting_id>/speakers')
def get_meeting_speakers(meeting_id):
    meeting = get_meeting_by_id(meeting_id)
    if not meeting:
        return jsonify({'error': 'Meeting not found'}), 404

    speaker_stats = meeting.get('speaker_stats', {})
    if not speaker_stats:
        return jsonify({'error': 'No speaker information available'}), 404

    return jsonify({
        'speaker_stats': speaker_stats,
        'unique_speakers': meeting.get('unique_speakers', 0),
        'speaker_summary': get_speaker_summary(speaker_stats)
    })


# --- NEW: Health check endpoint ---
@app.route('/health')
def health_check():
    """Health check endpoint to verify all components are working"""
    try:
        from .whisper_transcriber import check_whisperx_requirements
        whisperx_status = check_whisperx_requirements()

        return jsonify({
            'status': 'healthy',
            'whisperx_available': whisperx_status,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'hf_configured': bool(os.getenv('HUGGINGFACE_TOKEN'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

#
# if __name__ == '__main__':
#     print("Starting Flask server with WhisperX speaker identification...")
#     app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)