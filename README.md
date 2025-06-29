# 🚀 **KIU Smart Meeting Assistant**

> 🎙️ Transcribe. 🧠 Analyze. 🎨 Visualize. 🔍 Search. All in one intelligent assistant.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![OpenAI API](https://img.shields.io/badge/OpenAI-4_APIs-brightgreen.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)

---

## ✨ **Key Features**

| Feature                                | Description                                                                                              |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 😤 **Speaker Identification**          | Uses **WhisperX** to transcribe and label speakers with high accuracy.                                   |
| 🧠 **AI-Powered Insights**             | Extracts summaries, action items, and key decisions via **GPT-4 + Function Calling**.                    |
| 🎨 **Visual Summaries**                | Generates infographic-style images with **DALL·E 3** for at-a-glance recaps.                             |
| 🔍 **Hybrid Search**                   | Combines semantic search with keyword matching for ultra-precise results. Supports multilingual queries. |
| 💡 **Smart Suggestions**               | Recommends contextually relevant meetings based on semantic similarity.                                  |
| 🗓️ **Calendar & Task Sync** *(Bonus)* | Automatically syncs tasks with **Todoist** using detected action items.                                  |

---

## 🧱 **Architecture Overview**

```mermaid
graph TD
    A[💅️ Frontend (HTML/JS)] -->|REST API| B[🌐 Flask Backend]
    B --> C[⚙️ Background Worker Thread]
    C --> D[🎧 WhisperX Transcription]
    D --> E[🧠 GPT-4 Analysis]
    E --> F[🖼️ DALL·E 3 Visualization]
    E --> G[🔍 Embedding + Keyword Search]
```

* **Frontend:** Vanilla JavaScript SPA (Single Page Application)
* **Backend:** Flask (Async-supported)
* **Worker:** Async pipeline for processing large AI workloads

---

## 📦 **Installation Guide**

### 🔧 Prerequisites

* Python 3.10+
* `pip` & `venv`
* **FFmpeg** (Required by Whisper) – [Download](https://ffmpeg.org/download.html)
* Hugging Face account + token – [Get Token](https://huggingface.co/settings/tokens)

---

### 💠 Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd final

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with API keys
```

### 🚁 `.env` Example:

```dotenv
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf-...
```

> ✅ **Don’t forget** to accept model terms on Hugging Face:

* [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
* [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

---

## ▶️ **How to Run**

```bash
# From the root directory
python run.py
```

> The app will run at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ **Running Tests**

```bash
pytest
```

Runs test suite located in the `/tests` folder to verify core functionality.

---

## 🌍 **Future Enhancements**

* 📤 Integration with Google Calendar
* 🧑‍💼 User management & meeting permissions
* 📊 Analytics dashboard with meeting KPIs
* 🌐 Browser extension for in-call assistant

---

## 🙌 **Contributing**

Pull requests and issue reports are welcome!
Feel free to fork and create enhancements 🚀

---

## 📍 **License**

Licensed under the MIT License. See `LICENSE` file for more info.
