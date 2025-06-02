# Habr Topic Analysis FastAPI Service

This project provides a FastAPI-based backend for topic modeling, trending analysis, and interactive dashboarding of Habr.com content. It leverages BERTopic, sentence-transformers, and various statistical and visualization tools.

---

## 1. Prerequisites

- **Python version:** 3.8 or higher (recommended: 3.9+)
- **Hardware:** For best performance, a machine with a CUDA-capable GPU is recommended, but CPU-only is supported.
- **Disk space:** You need at least 10GB free (the main CSV and model files are large).
- **OS:** Windows, Linux, or MacOS.

---

## 2. Installation

### 2.1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 2.3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4. Download spaCy Russian model

```bash
python -m spacy download ru_core_news_lg
```

### 2.5. Download NLTK stopwords

The first run will attempt to download NLTK stopwords. To do it manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 3. Required Data Files

Place the following files in the project root (or as specified):

- `texts_with_topics.csv` (main data, **~5GB**)
- `bertopic_model_sber_sbert_REDUCED` (BERTopic model, **~3GB**)
- `topicmodel2305/topic_embeddings_sber.pkl` (topic embeddings, **~600KB**)
- `static/dashboard.html` (frontend dashboard, already present)

**Note:** These files are large. Ensure you have enough disk space and use a fast disk for best performance.

---

## 4. Running the Service

### 4.1. Start the FastAPI server

```bash
uvicorn service:app --reload
```

- The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- The interactive dashboard: [http://127.0.0.1:8000/static/dashboard.html](http://127.0.0.1:8000/static/dashboard.html)

### 4.2. API Endpoints

- `/analyze` (POST): Analyze a text and get topic/cluster info.
- `/api/topics`: List all topics.
- `/api/topic_examples/{topic_id}`: Example texts for a topic.
- `/api/topic_words/{topic_id}`: Top words for a topic.
- `/api/top_topics`: Top topics in the last 30 days.
- `/api/trending_posts`: Trending posts in the last 30 days.
- `/api/topic_trends`: Topic trends over the last year.
- `/api/trending_topics_over_time`: Trending topics by period.
- `/api/emerging_topics`: Emerging topics by frequency/views.
- `/api/emerging_topic_clusters`: Emerging topic clusters.
- `/api/topic_clusters`: Topic clusters by embedding similarity.
- `/api/topic_clusters_plot`: 2D plot of topic clusters.
- `/api/trending_topic_clusters`: Trending clusters by period.
- `/api/trending_topic_clusters_moving_avg`: Trending clusters by moving average.

See `service.py` for full details.

---

## 5. Troubleshooting

- **Large files:** If you are missing any required files, contact the project maintainer or use the provided scripts to generate them.
- **CUDA/CPU:** The service will use GPU if available, otherwise CPU.
- **NLTK errors:** If you see errors about missing stopwords or punkt, run the NLTK download commands above.
- **Port in use:** If port 8000 is busy, use `--port 8080` or another port:  
  `uvicorn service:app --reload --port 8080`

---

## 6. Updating/Regenerating Data

- To update or regenerate embeddings, topics, or clusters, use the scripts in the repo (e.g., `maketopic_embeddings.py`, `topic_modeling.py`).
- For new data, ensure it matches the format of `texts_with_topics.csv`.

---

## 7. Frontend Dashboard

- The dashboard is served at `/static/dashboard.html`.
- It uses Chart.js and fetches data from the FastAPI backend.
- No additional build steps are required for the frontend.

---

## 8. Example Usage

### Analyze a text (Python)

```python
import requests
text = "Ваш текст для анализа"
response = requests.post("http://127.0.0.1:8000/analyze", data=text.encode("utf-8"))
print(response.json())
```

---

## 9. File Structure

- `service.py` — FastAPI backend
- `static/dashboard.html` — Frontend dashboard
- `texts_with_topics.csv` — Main data
- `bertopic_model_sber_sbert_REDUCED` — BERTopic model
- `topicmodel2305/topic_embeddings_sber.pkl` — Topic embeddings
- `requirements.txt` — Python dependencies

---

## 10. Notes

- The first run may take several minutes to load models and data.
- For production, remove `--reload` and consider using a process manager.
- For large-scale use, consider optimizing data loading and using a database.

---

**If you encounter any issues, please open an issue or contact the maintainer.** 