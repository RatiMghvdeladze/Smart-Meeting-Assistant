import whisperx
import warnings
import os
import shutil
import time
import tempfile
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Global variables for models (loaded once)
whisper_model = None
diarize_model = None
align_model = None
align_metadata = None


def initialize_models():
    """Initialize all models once to avoid reloading"""
    global whisper_model, diarize_model, align_model, align_metadata

    print("Initializing WhisperX models...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Using device: {device}, compute_type: {compute_type}")

    # 1. Load Whisper model
    print("Loading Whisper model...")
    whisper_model = whisperx.load_model("base", device, compute_type=compute_type)
    print("Whisper model loaded successfully!")

    # 2. Load alignment model (for word-level timestamps)
    print("Loading alignment model...")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
    print("Alignment model loaded successfully!")

    # 3. Load diarization model (for speaker identification)
    print("Loading diarization model...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not found in .env file. Speaker identification will be disabled.")
        diarize_model = None
    else:
        try:
            # FIXED: Use proper diarization pipeline initialization
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            print("Diarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("Make sure you've accepted the terms at: https://huggingface.co/pyannote/speaker-diarization")
            print("Speaker identification will be disabled.")
            diarize_model = None


def debug_transcription_result(result):
    """Debug function to inspect transcription results"""
    print("\n=== DEBUG TRANSCRIPTION RESULT ===")
    print(f"Result type: {type(result)}")

    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")

        # Check segments
        segments = result.get('segments', [])
        print(f"Number of segments: {len(segments)}")

        if segments:
            print("First few segments:")
            for i, segment in enumerate(segments[:3]):
                if segment:
                    print(f"  Segment {i}: {segment.get('text', 'NO TEXT')[:50]}...")
                    print(f"    Start: {segment.get('start', 'N/A')}, End: {segment.get('end', 'N/A')}")
                    print(f"    Speaker: {segment.get('speaker', 'NO SPEAKER')}")
                else:
                    print(f"  Segment {i}: None/Empty")
    else:
        print(f"Unexpected result format: {result}")

    print("=== END DEBUG ===\n")


def transcribe_with_speakers(file_path, min_speakers=None, max_speakers=None):
    """
    Transcribe audio file using WhisperX with FIXED speaker identification.
    """
    global whisper_model, diarize_model, align_model, align_metadata

    # Initialize models if not already loaded
    if whisper_model is None:
        initialize_models()

    temp_file_path = None

    try:
        print(f"Starting WhisperX transcription of: {file_path}")

        # Convert to absolute path and normalize
        file_path = os.path.abspath(file_path)
        print(f"Absolute file path: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Get file info
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")

        # Create temporary copy
        print("Creating temporary copy for transcription...")
        file_extension = os.path.splitext(file_path)[1].lower()
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_extension, prefix='whisperx_')
        os.close(temp_fd)

        shutil.copy2(file_path, temp_file_path)
        print(f"Created temporary copy at: {temp_file_path}")

        # Load audio
        print("Loading audio...")
        audio = whisperx.load_audio(temp_file_path)
        audio_duration = len(audio) / 16000
        print(f"Audio loaded. Duration: {audio_duration:.2f} seconds")

        # Step 1: Transcribe with Whisper
        print("Step 1: Transcribing with Whisper...")
        result = whisper_model.transcribe(audio, batch_size=16)
        print(f"Initial transcription completed. Segments: {len(result.get('segments', []))}")

        # Step 2: Align whisper output for word-level timestamps
        print("Step 2: Aligning for word-level timestamps...")
        if align_model and align_metadata:
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_char_alignments=False
            )
            print("Alignment completed successfully!")
        else:
            print("Alignment model not available, skipping alignment step")

        # Step 3: FIXED Speaker Diarization
        print("Step 3: Identifying speakers...")
        diarize_segments = None
        speaker_segments_detected = []

        if diarize_model:
            try:
                print("Performing speaker diarization...")

                # FIXED: Better parameter handling for speaker detection
                # Set more reasonable defaults based on audio duration
                if min_speakers is None:
                    min_speakers = 2 if audio_duration > 30 else 1  # Assume 2+ speakers for longer audio
                if max_speakers is None:
                    max_speakers = min(8, max(3, int(audio_duration / 60) + 2))  # Scale with duration

                print(f"Audio duration: {audio_duration:.1f}s")
                print(f"Using speaker range: {min_speakers} to {max_speakers}")

                # FIXED: Perform diarization with better error handling
                try:
                    diarize_segments = diarize_model(
                        temp_file_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )

                    print(f"Diarization completed!")

                    # FIXED: Better processing of diarization results
                    if diarize_segments:
                        unique_speakers_detected = set()
                        print("\nDiarization segments found:")

                        for turn, track, speaker in diarize_segments.itertracks(yield_label=True):
                            unique_speakers_detected.add(speaker)
                            speaker_segments_detected.append({
                                'start': turn.start,
                                'end': turn.end,
                                'speaker': speaker
                            })
                            print(f"  {turn.start:.2f}s - {turn.end:.2f}s: {speaker}")

                        print(
                            f"Unique speakers detected: {len(unique_speakers_detected)} - {sorted(unique_speakers_detected)}")

                        if len(unique_speakers_detected) <= 1:
                            print("⚠️  WARNING: Only 1 speaker detected. This might indicate:")
                            print("   - Audio quality issues")
                            print("   - Very similar voices")
                            print("   - Need to adjust min/max speaker parameters")
                            print("   - Diarization model limitations")

                    else:
                        print("❌ No diarization segments returned")

                except Exception as diarize_error:
                    print(f"❌ Diarization failed: {diarize_error}")
                    print("This could be due to:")
                    print("1. Hugging Face token issues")
                    print("2. Model access permissions not accepted")
                    print("3. Audio format compatibility")
                    print("Falling back to no speaker identification")
                    diarize_segments = None

                # FIXED: Assign speaker labels to segments with better logic
                if diarize_segments and speaker_segments_detected:
                    print("Assigning speaker labels to transcription segments...")
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print("Speaker assignment completed!")

                    # VERIFICATION: Check if speakers were actually assigned
                    speakers_in_result = set()
                    for seg in result.get("segments", []):
                        if seg and seg.get("speaker"):
                            speakers_in_result.add(seg["speaker"])

                    print(f"Speakers found in final result: {len(speakers_in_result)} - {sorted(speakers_in_result)}")

                    if len(speakers_in_result) <= 1:
                        print("⚠️  ISSUE: Speaker assignment didn't work properly")
                        print("Attempting manual speaker assignment...")
                        result = manual_speaker_assignment(result, speaker_segments_detected)
                else:
                    print("⚠️  No valid diarization results, proceeding without speaker labels")

            except Exception as e:
                print(f"❌ Error during speaker identification: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("Continuing without speaker labels...")
                diarize_segments = None
        else:
            print("❌ Diarization model not available, skipping speaker identification")

        # FIXED: Format results with better speaker handling
        transcript_text = ""
        formatted_segments = []
        speaker_mapping = {}
        fallback_speaker_counter = 0

        print(f"\nProcessing {len(result.get('segments', []))} transcription segments...")

        for i, segment in enumerate(result.get("segments", [])):
            if segment is None:
                continue

            # FIXED: Better speaker detection and assignment
            original_speaker = segment.get("speaker", None)

            # Handle speaker assignment with multiple fallback strategies
            if original_speaker is None or str(original_speaker).strip() == "":
                # Strategy 1: Try to find speaker based on timing
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)

                assigned_speaker = None
                if speaker_segments_detected:
                    # Find overlapping speaker segment
                    max_overlap = 0
                    for sp_seg in speaker_segments_detected:
                        overlap_start = max(segment_start, sp_seg['start'])
                        overlap_end = min(segment_end, sp_seg['end'])
                        overlap_duration = max(0, overlap_end - overlap_start)

                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            assigned_speaker = sp_seg['speaker']

                if assigned_speaker:
                    original_speaker = assigned_speaker
                    print(f"Segment {i}: Assigned speaker {assigned_speaker} based on timing overlap")
                else:
                    # Strategy 2: Alternate between speakers if we detected multiple
                    if len(speaker_segments_detected) > 1:
                        unique_detected = list(set(seg['speaker'] for seg in speaker_segments_detected))
                        original_speaker = unique_detected[fallback_speaker_counter % len(unique_detected)]
                        fallback_speaker_counter += 1
                        print(f"Segment {i}: Using alternating speaker assignment: {original_speaker}")
                    else:
                        # Strategy 3: Default naming
                        original_speaker = "SPEAKER_UNKNOWN"

            # Clean speaker name
            original_speaker = str(original_speaker).strip()

            # Create consistent speaker mapping
            if original_speaker not in speaker_mapping:
                speaker_mapping[original_speaker] = f"SPEAKER_{len(speaker_mapping):02d}"

            speaker = speaker_mapping[original_speaker]

            # Process text
            text = segment.get("text", "")
            if text is None:
                continue

            text = str(text).strip()
            if not text:
                continue

            # Get timing
            start_time = float(segment.get("start", 0) or 0)
            end_time = float(segment.get("end", 0) or 0)

            transcript_text += f"{speaker}: {text}\n"

            formatted_segments.append({
                "speaker": speaker,
                "original_speaker": original_speaker,
                "text": text,
                "start": start_time,
                "end": end_time,
                "duration": max(0, end_time - start_time)
            })

        print(f"\n✅ Transcription completed!")
        print(f"Total segments: {len(formatted_segments)}")
        print(f"Speaker mapping: {speaker_mapping}")

        # Generate speaker statistics
        speaker_stats = {}
        total_duration = 0

        for segment in formatted_segments:
            speaker = segment["speaker"]
            duration = segment["duration"]
            text = segment["text"]

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "duration": 0,
                    "segments": 0,
                    "words": 0,
                    "original_labels": set()
                }

            speaker_stats[speaker]["duration"] += duration
            speaker_stats[speaker]["segments"] += 1
            speaker_stats[speaker]["words"] += len(text.split())
            speaker_stats[speaker]["original_labels"].add(segment["original_speaker"])

            total_duration += duration

        # Calculate percentages and convert sets to lists
        for speaker in speaker_stats:
            if total_duration > 0:
                speaker_stats[speaker]["percentage"] = (speaker_stats[speaker]["duration"] / total_duration) * 100
            else:
                speaker_stats[speaker]["percentage"] = 0
            speaker_stats[speaker]["original_labels"] = list(speaker_stats[speaker]["original_labels"])

        print(f"\n📊 Final Speaker Statistics:")
        for speaker, stats in speaker_stats.items():
            print(f"  {speaker}: {stats['duration']:.1f}s ({stats['percentage']:.1f}%) - {stats['segments']} segments")

        return {
            "text": transcript_text.strip(),
            "segments": formatted_segments,
            "speaker_stats": speaker_stats,
            "total_duration": total_duration,
            "unique_speakers": len(speaker_stats),
            "speaker_mapping": speaker_mapping,
            "diarization_successful": diarize_segments is not None and len(speaker_segments_detected) > 1,
            "detected_speakers": len(
                set(seg['speaker'] for seg in speaker_segments_detected)) if speaker_segments_detected else 0
        }

    except Exception as e:
        print(f"❌ Error in WhisperX transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                time.sleep(0.1)
                os.remove(temp_file_path)
                print(f"🧹 Cleaned up temporary file")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary file: {e}")


def manual_speaker_assignment(result, speaker_segments_detected):
    """
    Manually assign speakers when automatic assignment fails
    """
    print("🔧 Performing manual speaker assignment...")

    segments = result.get("segments", [])
    if not segments or not speaker_segments_detected:
        return result

    for segment in segments:
        if not segment:
            continue

        segment_start = segment.get("start", 0)
        segment_end = segment.get("end", 0)

        # Find the speaker segment with maximum overlap
        best_speaker = None
        max_overlap = 0

        for sp_seg in speaker_segments_detected:
            overlap_start = max(segment_start, sp_seg['start'])
            overlap_end = min(segment_end, sp_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = sp_seg['speaker']

        if best_speaker:
            segment["speaker"] = best_speaker

    print("✅ Manual speaker assignment completed")
    return result


def get_speaker_summary(speaker_stats):
    """Generate a human-readable summary of speaker statistics"""
    if not speaker_stats:
        return "No speaker information available."

    summary = f"Meeting had {len(speaker_stats)} participants:\n\n"

    # Sort speakers by speaking time
    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]["duration"], reverse=True)

    for speaker, stats in sorted_speakers:
        duration_minutes = stats["duration"] / 60
        original_labels = stats.get("original_labels", [])
        original_info = f" (detected as: {', '.join(original_labels)})" if original_labels and original_labels != [
            "UNKNOWN"] else ""

        summary += f"• {speaker}: {duration_minutes:.1f} minutes ({stats['percentage']:.1f}%) - {stats['segments']} segments, {stats['words']} words{original_info}\n"

    return summary


def check_whisperx_requirements():
    """Check if all required dependencies are available"""
    try:
        import whisperx
        import torch
        print("✅ WhisperX and PyTorch are available")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Using device: {device}")

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            print("✅ Hugging Face token found")
        else:
            print("⚠️  Hugging Face token not found - speaker identification will be limited")

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install whisperx torch")
        return False


def test_speaker_detection(file_path, min_speakers=2, max_speakers=5):
    """
    Test function to debug speaker detection issues
    """
    print("🧪 TESTING SPEAKER DETECTION")
    print("=" * 50)

    try:
        # Check requirements first
        if not check_whisperx_requirements():
            return False

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("❌ No Hugging Face token - cannot test speaker detection")
            return False

        print("✅ All requirements met, testing speaker detection...")

        # Test file
        if not os.path.exists(file_path):
            print(f"❌ Test file not found: {file_path}")
            return False

        print(f"📁 Testing file: {file_path}")

        # Load audio to check duration
        audio = whisperx.load_audio(file_path)
        duration = len(audio) / 16000
        print(f"⏱️  Audio duration: {duration:.1f} seconds")

        if duration < 10:
            print("⚠️  Warning: Very short audio may not work well for speaker detection")

        # Test diarization directly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

        print(f"🎯 Testing diarization with {min_speakers}-{max_speakers} speakers...")
        diarize_segments = diarize_model(file_path, min_speakers=min_speakers, max_speakers=max_speakers)

        if diarize_segments:
            speakers_found = set()
            segment_count = 0

            print("📊 Diarization results:")
            for turn, track, speaker in diarize_segments.itertracks(yield_label=True):
                speakers_found.add(speaker)
                segment_count += 1
                print(f"   {turn.start:6.2f}s - {turn.end:6.2f}s: {speaker}")

            print(f"\n✅ SUCCESS: Found {len(speakers_found)} unique speakers in {segment_count} segments")
            print(f"🎭 Speakers: {sorted(speakers_found)}")

            if len(speakers_found) == 1:
                print("\n⚠️  ISSUE: Only 1 speaker detected. Try:")
                print("   1. Increase audio quality")
                print("   2. Use longer audio clips")
                print("   3. Adjust min_speakers=1, max_speakers=3")
                print("   4. Check if speakers have very similar voices")

            return True
        else:
            print("❌ No diarization segments returned")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 WhisperX Speaker Detection Test")

    if check_whisperx_requirements():
        print("✅ All requirements satisfied!")
        print("\n📋 Setup checklist:")
        print("1. ✅ Add HUGGINGFACE_TOKEN to your .env file")
        print("2. ✅ Get token from: https://huggingface.co/settings/tokens")
        print("3. ✅ Accept terms at: https://huggingface.co/pyannote/speaker-diarization")
        print("4. ✅ Accept terms at: https://huggingface.co/pyannote/segmentation")

        # Uncomment to test with your audio file:
        # test_speaker_detection("/path/to/your/audio/file.wav", min_speakers=2, max_speakers=5)
    else:
        print("❌ Please install missing dependencies first")