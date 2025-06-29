from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ").strip()
        if not text: return None
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    if not words: return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


def create_embeddings_for_meeting(searchable_text):
    if not searchable_text: return []
    chunks = chunk_text(searchable_text)
    embeddings = []
    for chunk in chunks:
        embedding_vector = get_embedding(chunk)
        if embedding_vector:
            embeddings.append({"chunk": chunk, "embedding": embedding_vector})
    return embeddings


def keyword_search(query, meetings):
    results = []
    query_lower = query.lower()
    for meeting in meetings:
        search_texts = [meeting.get('summary', ''), meeting.get('transcript', '')]
        found_context = ""
        for text in search_texts:
            if query_lower in text.lower():
                sentences = re.split(r'(?<=[.?!])\s+', text)
                for sentence in sentences:
                    if query_lower in sentence.lower():
                        found_context = sentence.strip()
                        break
                if found_context: break
        if found_context:
            results.append({
                "id": meeting["id"], "title": meeting.get("title", "Untitled"),
                "created_at": meeting.get("created_at"), "summary": meeting.get("summary", ""),
                "similarity": 0.2, "best_chunk": f"...{found_context}...",
                "search_type": "keyword"
            })
    return results


def search_meetings(query, meetings):
    if not query or not meetings: return []
    query_embedding = get_embedding(query)
    semantic_matches = {}
    if query_embedding:
        for meeting in meetings:
            if "chunk_embeddings" in meeting and meeting["chunk_embeddings"]:
                best_similarity, best_chunk_text = -1.0, ""
                for chunk_data in meeting["chunk_embeddings"]:
                    sim = cosine_similarity([query_embedding], [chunk_data["embedding"]])[0][0]
                    if sim > best_similarity:
                        best_similarity, best_chunk_text = sim, chunk_data["chunk"]
                if best_similarity > 0.4:
                    semantic_matches[meeting["id"]] = {
                        "id": meeting["id"], "title": meeting.get("title", "Untitled"),
                        "created_at": meeting.get("created_at"), "summary": meeting.get("summary", ""),
                        "similarity": round(best_similarity, 3), "best_chunk": best_chunk_text,
                        "search_type": "semantic"
                    }
    keyword_results = keyword_search(query, meetings)
    final_results = semantic_matches
    for kw_result in keyword_results:
        if kw_result["id"] not in final_results:
            final_results[kw_result["id"]] = kw_result
        else:
            final_results[kw_result["id"]]['similarity'] += 0.1
            final_results[kw_result["id"]]['search_type'] += "+keyword"
    return sorted(list(final_results.values()), key=lambda x: x["similarity"], reverse=True)


# --- NEW FUNCTION FOR SIMILARITY-BASED RECOMMENDATIONS ---
def find_similar_meetings(source_meeting_id, all_meetings, top_n=3):
    """
    Finds meetings that are semantically similar to a given source meeting.
    """
    source_meeting = None
    other_meetings = []
    for m in all_meetings:
        if m.get('id') == source_meeting_id:
            source_meeting = m
        else:
            other_meetings.append(m)

    # Ensure the source meeting and its embedding exist
    if not source_meeting or "chunk_embeddings" not in source_meeting or not source_meeting["chunk_embeddings"]:
        return []

    # For simplicity, we'll use the embedding of the first chunk as the representative vector.
    # A more advanced method would average all chunk embeddings.
    source_vector = source_meeting["chunk_embeddings"][0]["embedding"]

    recommendations = []
    for meeting in other_meetings:
        if "chunk_embeddings" in meeting and meeting["chunk_embeddings"]:
            target_vector = meeting["chunk_embeddings"][0]["embedding"]
            similarity = cosine_similarity([source_vector], [target_vector])[0][0]

            if similarity > 0.6:  # Similarity threshold for recommendations
                recommendations.append({
                    "id": meeting["id"],
                    "title": meeting["title"],
                    "summary": meeting.get("summary", ""),
                    "similarity": round(similarity, 2)
                })

    # Sort by similarity and return the top N results
    recommendations.sort(key=lambda x: x["similarity"], reverse=True)
    return recommendations[:top_n]