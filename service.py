from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder
from tqdm.auto import tqdm
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objs as go

# === Load models and data at startup ===

# Load spaCy model
nlp = spacy.load("ru_core_news_lg")

# Load stopwords
russian_stopwords = set(stopwords.words('russian'))
english_stopwords = set(stopwords.words('english'))
all_stopwords = russian_stopwords.union(english_stopwords)
custom_stopwords = {
    'это', 'этот', 'эта', 'эти', 'этот', 'этих', 'этим', 'этими',
    'который', 'которая', 'которое', 'которые',
    'кто', 'что', 'какой', 'какая', 'какое', 'какие',
    'весь', 'вся', 'всё', 'все',
    'свой', 'своя', 'своё', 'свои',
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'дан', 'работ', 'котор', 'одн', 'друг', 'нужн', 'кажд', 'прост', 'наш', 'систем'
}
all_stopwords = all_stopwords.union(custom_stopwords)

# Load SentenceTransformer model
device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer("ai-forever/sbert_large_nlu_ru", device=device, truncate_dim=512)

# Load BERTopic model
topic_model = BERTopic.load('bertopic_model_sber_sbert_REDUCED')

# Patch torch.load to force map_location='cpu' for BERTopic loading
_torch_load_old = torch.load
def _torch_load_cpu(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return _torch_load_old(*args, **kwargs)
torch.load = _torch_load_cpu

# Set custom embedding model for BERTopic
class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model):
        self.model = model
    def embed(self, documents, verbose=False):
        return self.model.encode(documents, show_progress_bar=verbose)
    def embed_documents(self, documents, verbose=False):
        return self.embed(documents, verbose=verbose)
    def embed_queries(self, queries, verbose=False):
        return self.embed(queries, verbose=verbose)
topic_model.embedding_model = SentenceTransformerEmbedder(sentence_model)

# Load main CSV data
df_top_texts = pd.read_csv("texts_with_topics.csv")

# Load precomputed topic embeddings
with open('topicmodel2305/topic_embeddings_sber.pkl', 'rb') as f:
    topic_embeddings = pickle.load(f)

# Compute top_topics and topic_rank as before, for leaderboard/explanation
N = 10
avg_views_by_topic = df_top_texts.groupby('topic')['views'].mean().sort_values(ascending=False)
top_topics = avg_views_by_topic.head(N).index.tolist()
topic_rank = {topic_id: rank+1 for rank, topic_id in enumerate(top_topics)}

def filter_tokens(text):
    words = word_tokenize(text)
    return " ".join([
        word for word in words
        if not re.match(r'^[\d\.,]+$', word) and word.lower() not in all_stopwords and len(word) >= 2
    ])

def preprocess_single_text(text, nlp):
    text = str(text).lower()
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    processed = filter_tokens(lemmatized)
    return processed

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# --- Helper: Moving Average Trending Clusters ---
def get_trending_clusters_moving_average(df, cluster_col, window_days=30, n_windows=3, n_top=5):
    """
    Compute trending clusters using a moving average of mean views over the last n_windows (each window_days long).
    Returns a set of cluster IDs considered trending.
    """
    df = df.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    if df.empty:
        return set()
    max_date = df['time_published'].max()
    cluster_ma_dict = {}
    for i in range(n_windows):
        window_end = max_date - pd.Timedelta(days=window_days * i)
        window_start = window_end - pd.Timedelta(days=window_days)
        window_df = df[(df['time_published'] > window_start) & (df['time_published'] <= window_end)]
        cluster_means = window_df.groupby(cluster_col)['views'].mean()
        for cid in cluster_means.index:
            if cid not in cluster_ma_dict:
                cluster_ma_dict[cid] = []
            cluster_ma_dict[cid].append(cluster_means[cid])
    # Only keep clusters with enough history
    cluster_ma = {cid: np.mean(vals) for cid, vals in cluster_ma_dict.items() if len(vals) == n_windows}
    if not cluster_ma:
        return set()
    trending_clusters = set(pd.Series(cluster_ma).sort_values(ascending=False).head(n_top).index)
    return trending_clusters

@app.post("/analyze")
async def analyze(request: Request, cluster_threshold: float = Query(0.2)):
    user_text = await request.body()
    user_text = user_text.decode("utf-8")
    user_text_processed = preprocess_single_text(user_text, nlp)
    user_emb = sentence_model.encode([user_text_processed])[0]

    # --- Only Cosine Similarity ---
    similarities = []
    for topic_id, topic_emb in topic_embeddings.items():
        cos_sim = cosine_similarity([user_emb], [topic_emb])[0][0]
        similarities.append((topic_id, cos_sim))

    # Sort and get best
    sim_list_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    best_topic = int(sim_list_sorted[0][0])
    best_score = float(sim_list_sorted[0][1])
    top3_by_metric = [{"topic": int(tid), "score": float(score)} for tid, score in sim_list_sorted[:3]]

    # Assign topic using BERTopic model's transform
    model_assigned_topic, _ = topic_model.transform([user_text_processed])
    model_assigned_topic = int(model_assigned_topic[0]) if hasattr(model_assigned_topic, '__getitem__') else int(model_assigned_topic)

    # Get top words for model-assigned topic
    if model_assigned_topic != -1:
        model_topic_words_scores = topic_model.get_topic(model_assigned_topic)
        best_topic_model_top_words = [w for w, _ in model_topic_words_scores[:10]] if model_topic_words_scores else []
    else:
        best_topic_model_top_words = []

    topic_words_scores = topic_model.get_topic(best_topic)
    topic_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
    topic_df = df_top_texts[df_top_texts['topic'] == best_topic]
    avg_views = topic_df['views'].mean()
    avg_comments = topic_df['comments_count'].mean()

    # Compute similarity of best_topic to all other topics and include top words (cosine only)
    best_topic_emb = topic_embeddings[best_topic]
    topic_connections = []
    for topic_id, emb in topic_embeddings.items():
        if topic_id == best_topic:
            continue
        sim = cosine_similarity([best_topic_emb], [emb])[0][0]
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words_conn = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        topic_connections.append({
            "topic_id": int(topic_id),
            "similarity": float(sim),
            "top_words": top_words_conn
        })
    topic_connections = sorted(topic_connections, key=lambda x: x["similarity"], reverse=True)[:10]

    # --- Most similar topics to input text (cosine) ---
    most_similar_topics = []
    top5_topic_ids = []
    for topic_id, sim in sim_list_sorted[:5]:
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        most_similar_topics.append({
            "topic_id": int(topic_id),
            "similarity": float(sim),
            "top_words": top_words
        })
        top5_topic_ids.append(topic_id)

    # --- Predicted views based on top 5 topics ---
    avg_views_list = []
    for tid in top5_topic_ids:
        topic_df = df_top_texts[df_top_texts['topic'] == tid]
        avg_views_list.append(topic_df['views'].mean() if not topic_df.empty else 0)
    predicted_views_top5 = float(np.mean(avg_views_list)) if avg_views_list else 0.0

    # --- EXPLANATION: for top 5 most similar topics ---
    explanation_lines = []
    explanation_lines.append("Why are these topics considered similar to your input?")
    lemm_input_set = set(user_text_processed.lower().split())
    topic_infos = []
    for i, t in enumerate(most_similar_topics):
        topic_id = t["topic_id"]
        sim = t["similarity"]
        top_words = t["top_words"]
        matched_words = [w for w in top_words if w in lemm_input_set]
        topic_df = df_top_texts[df_top_texts['topic'] == topic_id]
        avg_views = topic_df['views'].mean() if not topic_df.empty else 0
        topic_infos.append({
            "idx": i,
            "topic_id": topic_id,
            "sim": sim,
            "top_words": top_words,
            "matched_words": matched_words,
            "avg_views": avg_views
        })
    prioritized = [info for info in topic_infos if info["matched_words"]]
    prioritized_topic = None
    if prioritized:
        prioritized_topic = max(prioritized, key=lambda x: x["avg_views"])
    for info in topic_infos:
        if info is prioritized_topic:
            explanation_lines.append(f"#{info['idx']+1} Topic {info['topic_id']} (similarity={info['sim']:.3f}, avg views={info['avg_views']:.1f}): Shares words with your input: {', '.join(info['matched_words'])} <b>[PRIORITIZED: highest avg views among shared words]</b>.")
        elif info["matched_words"]:
            explanation_lines.append(f"#{info['idx']+1} Topic {info['topic_id']} (similarity={info['sim']:.3f}, avg views={info['avg_views']:.1f}): Shares words with your input: {', '.join(info['matched_words'])}.")
        else:
            explanation_lines.append(f"#{info['idx']+1} Topic {info['topic_id']} (similarity={info['sim']:.3f}, avg views={info['avg_views']:.1f}): No shared top words, but high semantic similarity.")
    if prioritized_topic:
        explanation_lines.append(f"\nAmong the most similar topics, Topic {prioritized_topic['topic_id']} is prioritized because it shares words with your input and has the highest average views.")
    elif prioritized:
        explanation_lines.append(f"\nAmong the most similar topics, those with shared words are prioritized in determining the input text's topic.")
    else:
        explanation_lines.append(f"\nNo most similar topics share top words with your input; semantic similarity is used.")
    explanation = "\n".join(explanation_lines)

    # Decide on the final best topic
    if best_topic == -1 and model_assigned_topic != -1:
        final_best_topic = model_assigned_topic
    elif model_assigned_topic == -1 and best_topic != -1:
        final_best_topic = best_topic
    elif best_topic == -1 and model_assigned_topic == -1:
        final_best_topic = None
    else:
        final_best_topic = best_topic

    # Cluster topics
    topic_ids = list(topic_embeddings.keys())
    embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=cluster_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    topic_to_cluster = {tid: int(label) for tid, label in zip(topic_ids, labels)}
    assigned_cluster = topic_to_cluster.get(int(final_best_topic), None)

    # --- Assigned cluster topics and top words ---
    assigned_cluster_topics = []
    if assigned_cluster is not None:
        # Find all topics in the assigned cluster
        cluster_topic_ids = [tid for tid, label in topic_to_cluster.items() if label == assigned_cluster]
        for tid in cluster_topic_ids:
            top_words = [w for w, _ in topic_model.get_topic(tid)[:10]]
            assigned_cluster_topics.append({
                "topic_id": int(tid),
                "top_words": top_words
            })

    # --- Avg views/comments for assigned cluster (all time, not just recent) ---
    if assigned_cluster is not None:
        cluster_df_all = df_top_texts[df_top_texts['topic'].isin(cluster_topic_ids)]
        avg_views = cluster_df_all['views'].mean() if not cluster_df_all.empty else 0
        avg_comments = cluster_df_all['comments_count'].mean() if not cluster_df_all.empty else 0
    else:
        avg_views = 0
        avg_comments = 0

    # --- Trending clusters (moving average of last 3 windows, top 5 by avg views) ---
    # Assign cluster_id to all texts
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    topic_to_cluster_for_df = {int(tid): int(label) for tid, label in zip(topic_ids, labels)}
    df['cluster_id'] = df['topic'].map(topic_to_cluster_for_df)
    trending_clusters = get_trending_clusters_moving_average(df, 'cluster_id', window_days=30, n_windows=3, n_top=5)
    is_trending = assigned_cluster in trending_clusters if assigned_cluster is not None else False

    # --- Prepare trending_clusters info for response (top 5 by moving average) ---
    trending_clusters_info = []
    for cid in trending_clusters:
        tids = [tid for tid, label in topic_to_cluster.items() if label == cid]
        cluster_df = df[df['cluster_id'] == cid]
        avg_views_trend = cluster_df['views'].mean() if not cluster_df.empty else 0
        trending_clusters_info.append({
            "cluster_id": int(cid),
            "avg_views": float(avg_views_trend),
            "topics": [
                {
                    "topic_id": int(tid),
                    "top_words": [w for w, _ in topic_model.get_topic(tid)[:10]]
                } for tid in tids
            ]
        })
    trending_clusters_info = sorted(trending_clusters_info, key=lambda x: x["avg_views"], reverse=True)

    return {
        "best_topic_semantic": int(best_topic) if best_topic != -1 else -1,
        "similarity": float(best_score),
        "best_topic_model": int(model_assigned_topic) if model_assigned_topic != -1 else -1,
        "final_best_topic": int(final_best_topic) if final_best_topic is not None else None,
        "best_topic_model_top_words": best_topic_model_top_words,
        "top_words": topic_words,
        "avg_views": float(avg_views),
        "avg_comments": float(avg_comments),
        "topic_rank": topic_rank.get(best_topic, None),
        "explanation": explanation,
        "lemmatized_text": user_text_processed,
        "topic_connections": topic_connections,
        "most_similar_topics": most_similar_topics,
        "predicted_views_top5": predicted_views_top5,
        "assigned_cluster": assigned_cluster,
        "is_cluster_trending": is_trending,
        "cluster_threshold": cluster_threshold,
        "assigned_cluster_topics": assigned_cluster_topics,
        "trending_clusters": trending_clusters_info
    }

@app.get("/api/topics")
def get_topics():
    # topic_info should be a DataFrame with columns: Topic, Count, Name, etc.
    topic_info = topic_model.get_topic_info()
    topics = []
    for _, row in topic_info.iterrows():
        topic_id = int(row['Topic'])
        # Do NOT skip outlier topic (-1)
        words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in words_scores[:10]] if words_scores else []
        topics.append({
            "topic_id": topic_id,
            "count": int(row['Count']),
            "top_words": top_words
        })
    return JSONResponse(content=topics)

@app.get("/api/topic_examples/{topic_id}")
def get_topic_examples(topic_id: int):
    # Return example texts for a topic (from your df_top_texts or similar)
    examples = df_top_texts[df_top_texts['topic'] == topic_id]['text'].head(10).tolist()
    return JSONResponse(content=examples)

@app.get("/api/topic_words/{topic_id}")
def get_topic_words(topic_id: int):
    try:
        topic_id = int(topic_id)
    except Exception:
        return JSONResponse(content=[])
    words_scores = topic_model.get_topic(topic_id)
    if words_scores is None:
        return JSONResponse(content=[])
    return JSONResponse(content=words_scores)

@app.get("/api/top_topics")
def get_top_topics(n: int = 10):
    # Filter for last 30 days relative to latest post
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    if df['time_published'].dt.tz is not None:
        df['time_published'] = df['time_published'].dt.tz_convert(None)
    df = df[~df['time_published'].isna()]
    if df.empty:
        return JSONResponse(content={
            "date_range": None,
            "topics": [],
            "message": "No posts available in the data."
        })
    max_date = df['time_published'].max()
    cutoff = max_date - pd.Timedelta(days=30)
    recent_df = df[(df['time_published'] > cutoff) & (df['time_published'] <= max_date)]
    if recent_df.empty:
        return JSONResponse(content={
            "date_range": {
                "min": cutoff.isoformat() if pd.notna(cutoff) else None,
                "max": max_date.isoformat() if pd.notna(max_date) else None
            },
            "topics": [],
            "message": "No posts in the last 30 days of available data."
        })
    avg_views_by_topic = recent_df.groupby('topic')['views'].mean().sort_values(ascending=False)
    top_topics = avg_views_by_topic.head(n).index.tolist()
    result = []
    for topic_id in top_topics:
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        avg_views = avg_views_by_topic[topic_id]
        result.append({
            "topic_id": int(topic_id),
            "avg_views": float(avg_views),
            "top_words": top_words
        })
    min_date = recent_df['time_published'].min()
    max_date_actual = recent_df['time_published'].max()
    return JSONResponse(content={
        "date_range": {
            "min": min_date.isoformat() if pd.notna(min_date) else None,
            "max": max_date_actual.isoformat() if pd.notna(max_date_actual) else None
        },
        "topics": result,
        "message": None
    })

@app.get("/api/topic_views_leaderboard/{topic_id}")
def get_topic_views_leaderboard(topic_id: int, window: int = 9):
    # Compute average views by topic
    avg_views_by_topic = df_top_texts.groupby('topic')['views'].mean().sort_values(ascending=False)
    leaderboard = avg_views_by_topic.reset_index().rename(columns={"topic": "topic_id", "views": "avg_views"})
    leaderboard["topic_id"] = leaderboard["topic_id"].astype(int)
    # Find the rank (0-based)
    try:
        rank = leaderboard[leaderboard["topic_id"] == topic_id].index[0]
    except IndexError:
        return JSONResponse(content={"error": "Topic not found in leaderboard."}, status_code=404)
    # Compute window
    half = window // 2
    start = max(rank - half, 0)
    end = min(start + window, len(leaderboard))
    start = max(end - window, 0)  # adjust start if near the end
    window_topics = leaderboard.iloc[start:end]
    result = []
    for _, row in window_topics.iterrows():
        tid = int(row["topic_id"])
        topic_words_scores = topic_model.get_topic(tid)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        result.append({
            "topic_id": tid,
            "avg_views": float(row["avg_views"]),
            "top_words": top_words
        })
    return JSONResponse(content={
        "window": result,
        "selected_rank": int(rank) + 1,  # 1-based rank
        "total_topics": int(len(leaderboard))
    })

@app.get("/api/trending_posts")
def get_trending_posts(n: int = 5):
    """
    Get trending posts for the last 30 days, sorted by views.
    Query parameters:
        n (int): Number of posts to return (default 5). Increase this value to get more posts.
    """
    # Ensure 'time_published' is datetime and timezone-naive
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    # Remove timezone info if present
    if df['time_published'].dt.tz is not None:
        df['time_published'] = df['time_published'].dt.tz_convert(None)
    # Remove rows with all-NaT dates
    df = df[~df['time_published'].isna()]
    if df.empty:
        return JSONResponse(content={
            "date_range": None,
            "posts": [],
            "message": "No trending posts available in the data."
        })
    max_date = df['time_published'].max()
    cutoff = max_date - pd.Timedelta(days=30)
    recent_df = df[(df['time_published'] > cutoff) & (df['time_published'] <= max_date)]
    if recent_df.empty:
        return JSONResponse(content={
            "date_range": {
                "min": cutoff.isoformat() if pd.notna(cutoff) else None,
                "max": max_date.isoformat() if pd.notna(max_date) else None
            },
            "posts": [],
            "message": "No posts in the last 30 days of available data."
        })
    recent_df = recent_df.sort_values('views', ascending=False).head(n)
    min_date = recent_df['time_published'].min()
    max_date_actual = recent_df['time_published'].max()
    result = [
        {
            "id": row.get('id', None),
            "date": row['time_published'].isoformat() if not pd.isna(row['time_published']) else None,
            "topic": int(row['topic']) if not pd.isna(row['topic']) else None,
            "views": float(row['views']) if not pd.isna(row['views']) else None,
            "comments_count": int(row['comments_count']) if not pd.isna(row['comments_count']) else None,
            "text": row['text'] if 'text' in row else None,
            "text_spacy": row['text_spacy'] if 'text_spacy' in row else None,
            "hubs": row['hubs'] if 'hubs' in row and pd.notna(row['hubs']) else None
        }
        for _, row in recent_df.iterrows()
    ]
    return JSONResponse(content={
        "date_range": {
            "min": min_date.isoformat() if pd.notna(min_date) else None,
            "max": max_date_actual.isoformat() if pd.notna(max_date_actual) else None
        },
        "posts": result,
        "message": None
    })

@app.get("/api/topic_trends")
def get_topic_trends():
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    if df.empty:
        return JSONResponse(content={"message": "No data available."})
    # Filter to last year
    max_date = df['time_published'].max()
    min_date = max_date - pd.Timedelta(days=365)
    df = df[(df['time_published'] > min_date) & (df['time_published'] <= max_date)]
    if df.empty:
        return JSONResponse(content={"message": "No data in the last year."})
    # Create month column
    df['month'] = df['time_published'].dt.to_period('M').astype(str)
    # Group by topic and month, compute avg views
    grouped = df.groupby(['topic', 'month'])['views'].mean().reset_index()
    # Pivot to list of dicts per topic
    topic_trends = {}
    for _, row in grouped.iterrows():
        tid = int(row['topic'])
        if tid not in topic_trends:
            topic_trends[tid] = []
        topic_trends[tid].append({
            "month": row['month'],
            "avg_views": float(row['views'])
        })
    # Compute change in avg_views for each topic over the year
    topic_changes = []
    for tid, trend in topic_trends.items():
        if len(trend) < 2:
            continue
        change = trend[-1]['avg_views'] - trend[0]['avg_views']
        topic_changes.append((tid, change))
    # Select top 10 topics by absolute change
    top_n = 10
    top_topics = sorted(topic_changes, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_topic_ids = set(tid for tid, _ in top_topics)
    # Prepare result for only these topics
    result = [
        {"topic_id": tid, "trend": topic_trends[tid]} for tid in top_topic_ids
    ]
    return JSONResponse(content=result)

@app.get("/api/trending_topics_over_time")
def get_trending_topics_over_time(period: str = Query('M', description="Time period: 'M' for month, 'Q' for quarter, 'W' for week"), top_n: int = Query(10, description="Number of top trending topics to return")):
    """
    Identify trending topics over time based on change in average views per topic per period.
    Returns the top topics with the largest positive change (momentum) in the latest period.
    """
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    df['period'] = df['time_published'].dt.to_period(period)
    topic_period_views = (
        df.groupby(['topic', 'period'])['views']
        .mean()
        .reset_index()
        .sort_values(['topic', 'period'])
    )
    topic_period_views['prev_views'] = (
        topic_period_views.groupby('topic')['views'].shift(1)
    )
    topic_period_views['delta_views'] = (
        topic_period_views['views'] - topic_period_views['prev_views']
    )
    latest_period = topic_period_views['period'].max()
    recent_trends = topic_period_views[topic_period_views['period'] == latest_period]
    trending_topics = recent_trends.sort_values('delta_views', ascending=False)
    # Add top words for each topic
    result = []
    for _, row in trending_topics.head(top_n).iterrows():
        topic_id = int(row['topic'])
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        result.append({
            "topic_id": topic_id,
            "avg_views": float(row['views']),
            "delta_views": float(row['delta_views']) if not pd.isna(row['delta_views']) else None,
            "top_words": top_words
        })
    return JSONResponse(content={
        "period": str(latest_period),
        "topics": result
    })

@app.get("/api/emerging_topics")
def get_emerging_topics(
    window_days: int = Query(30, description="Size of the recent and baseline window in days"),
    min_baseline_count: int = Query(5, description="Minimum number of posts in baseline window to consider topic"),
    top_n: int = Query(10, description="Number of emerging topics to return")
):
    """
    Detect emerging (spiking) topics by comparing their frequency and average views in the most recent window vs. a baseline window.
    Returns topics with the largest increase in frequency or average views.
    """
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    if df.empty:
        return JSONResponse(content={"topics": [], "message": "No data available."})
    max_date = df['time_published'].max()
    recent_start = max_date - pd.Timedelta(days=window_days)
    baseline_start = recent_start - pd.Timedelta(days=window_days)
    # Recent and baseline windows
    recent_df = df[(df['time_published'] > recent_start) & (df['time_published'] <= max_date)]
    baseline_df = df[(df['time_published'] > baseline_start) & (df['time_published'] <= recent_start)]
    # Frequency and avg views per topic
    recent_stats = recent_df.groupby('topic').agg(recent_count=('id', 'count'), recent_avg_views=('views', 'mean')).reset_index()
    baseline_stats = baseline_df.groupby('topic').agg(baseline_count=('id', 'count'), baseline_avg_views=('views', 'mean')).reset_index()
    merged = pd.merge(recent_stats, baseline_stats, on='topic', how='outer').fillna(0)
    # Calculate increases
    merged['count_increase'] = merged['recent_count'] - merged['baseline_count']
    merged['count_ratio'] = (merged['recent_count'] + 1) / (merged['baseline_count'] + 1)
    merged['avg_views_increase'] = merged['recent_avg_views'] - merged['baseline_avg_views']
    merged['avg_views_ratio'] = (merged['recent_avg_views'] + 1) / (merged['baseline_avg_views'] + 1)
    # Filter topics with enough baseline data
    filtered = merged[merged['baseline_count'] >= min_baseline_count]
    # Sort by count_ratio and avg_views_ratio
    filtered = filtered.sort_values(['count_ratio', 'avg_views_ratio'], ascending=False)
    # Prepare result
    result = []
    for _, row in filtered.head(top_n).iterrows():
        topic_id = int(row['topic'])
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        result.append({
            "topic_id": topic_id,
            "recent_count": int(row['recent_count']),
            "baseline_count": int(row['baseline_count']),
            "count_ratio": float(row['count_ratio']),
            "recent_avg_views": float(row['recent_avg_views']),
            "baseline_avg_views": float(row['baseline_avg_views']),
            "avg_views_ratio": float(row['avg_views_ratio']),
            "top_words": top_words
        })
    return JSONResponse(content={
        "window_days": window_days,
        "recent_start": recent_start.isoformat() if pd.notna(recent_start) else None,
        "recent_end": max_date.isoformat() if pd.notna(max_date) else None,
        "baseline_start": baseline_start.isoformat() if pd.notna(baseline_start) else None,
        "baseline_end": recent_start.isoformat() if pd.notna(recent_start) else None,
        "topics": result
    })

@app.get("/api/emerging_topic_clusters")
def get_emerging_topic_clusters(
    window_days: int = Query(30, description="Size of the recent and baseline window in days"),
    min_baseline_count: int = Query(5, description="Minimum number of posts in baseline window to consider topic"),
    top_n: int = Query(10, description="Number of emerging topic clusters to return"),
    n_examples: int = Query(3, description="Number of example texts to return per topic")
):
    """
    For each topic (cluster), compare its frequency and average views in the recent window vs. baseline window.
    For emerging topics, return example texts, top words, and statistics.
    """
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    if df.empty:
        return JSONResponse(content={"topics": [], "message": "No data available."})
    max_date = df['time_published'].max()
    recent_start = max_date - pd.Timedelta(days=window_days)
    baseline_start = recent_start - pd.Timedelta(days=window_days)
    # Recent and baseline windows
    recent_df = df[(df['time_published'] > recent_start) & (df['time_published'] <= max_date)]
    baseline_df = df[(df['time_published'] > baseline_start) & (df['time_published'] <= recent_start)]
    # Frequency and avg views per topic
    recent_stats = recent_df.groupby('topic').agg(recent_count=('id', 'count'), recent_avg_views=('views', 'mean')).reset_index()
    baseline_stats = baseline_df.groupby('topic').agg(baseline_count=('id', 'count'), baseline_avg_views=('views', 'mean')).reset_index()
    merged = pd.merge(recent_stats, baseline_stats, on='topic', how='outer').fillna(0)
    # Calculate increases
    merged['count_increase'] = merged['recent_count'] - merged['baseline_count']
    merged['count_ratio'] = (merged['recent_count'] + 1) / (merged['baseline_count'] + 1)
    merged['avg_views_increase'] = merged['recent_avg_views'] - merged['baseline_avg_views']
    merged['avg_views_ratio'] = (merged['recent_avg_views'] + 1) / (merged['baseline_avg_views'] + 1)
    # Filter topics with enough baseline data
    filtered = merged[merged['baseline_count'] >= min_baseline_count]
    # Sort by count_ratio and avg_views_ratio
    filtered = filtered.sort_values(['count_ratio', 'avg_views_ratio'], ascending=False)
    # Prepare result with example texts
    result = []
    for _, row in filtered.head(top_n).iterrows():
        topic_id = int(row['topic'])
        topic_words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
        # Example texts from recent window
        examples = recent_df[recent_df['topic'] == topic_id]['text'].head(n_examples).tolist()
        result.append({
            "topic_id": topic_id,
            "recent_count": int(row['recent_count']),
            "baseline_count": int(row['baseline_count']),
            "count_ratio": float(row['count_ratio']),
            "recent_avg_views": float(row['recent_avg_views']),
            "baseline_avg_views": float(row['baseline_avg_views']),
            "avg_views_ratio": float(row['avg_views_ratio']),
            "top_words": top_words,
            "examples": examples
        })
    return JSONResponse(content={
        "window_days": window_days,
        "recent_start": recent_start.isoformat() if pd.notna(recent_start) else None,
        "recent_end": max_date.isoformat() if pd.notna(max_date) else None,
        "baseline_start": baseline_start.isoformat() if pd.notna(baseline_start) else None,
        "baseline_end": recent_start.isoformat() if pd.notna(recent_start) else None,
        "topics": result
    })

@app.get("/api/topic_clusters")
def get_topic_clusters(distance_threshold: float = Query(0.2, description="Distance threshold for clustering (1-cosine similarity), lower means tighter clusters")):
    """
    Cluster topics based on cosine similarity of their embeddings using Agglomerative Clustering.
    Returns clusters with their member topic IDs, top words, total texts count, and average views.
    """
    # Prepare topic embeddings
    topic_ids = list(topic_embeddings.keys())
    embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    # Convert similarity to distance
    distance_matrix = 1 - sim_matrix
    # Perform clustering
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    # Group topics by cluster
    clusters = {}
    for topic_id, label in zip(topic_ids, labels):
        clusters.setdefault(label, []).append(topic_id)
    # Prepare result
    result = []
    for label, topic_list in clusters.items():
        # Calculate total texts and avg views for the cluster
        cluster_df = df_top_texts[df_top_texts['topic'].isin(topic_list)]
        texts_count = int(cluster_df.shape[0])
        avg_views = float(cluster_df['views'].mean()) if not cluster_df.empty else 0.0
        cluster_info = {
            "cluster_id": int(label),
            "topics": [
                {
                    "topic_id": int(tid),
                    "top_words": [w for w, _ in topic_model.get_topic(tid)[:10]]
                }
                for tid in topic_list
            ],
            "texts_count": texts_count,
            "avg_views": avg_views
        }
        result.append(cluster_info)
    return JSONResponse(content={"distance_threshold": distance_threshold, "clusters": result})

@app.get("/api/topic_clusters_plot")
def get_topic_clusters_plot(distance_threshold: float = Query(0.2, description="Distance threshold for clustering (1-cosine similarity), lower means tighter clusters"), method: str = Query('umap', description="Dimensionality reduction: 'umap' or 'tsne'")):
    # Prepare topic embeddings and clustering (reuse logic from /api/topic_clusters)
    topic_ids = list(topic_embeddings.keys())
    embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    # 2D projection
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', s=100, alpha=0.8, edgecolors='k')
    for i, tid in enumerate(topic_ids):
        top_words = topic_model.get_topic(tid)
        label = f"{tid}: {top_words[0][0] if top_words else ''}"
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=8, alpha=0.7)
    plt.title("Topic Clusters (2D projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/trending_topic_clusters")
def get_trending_topic_clusters(
    distance_threshold: float = Query(0.2, description="Distance threshold for clustering (1-cosine similarity)"),
    period: str = Query('M', description="Time period: 'M' for month, '6M' for 6 months, 'A' for year"),
    top_n: int = Query(10, description="Number of top trending clusters to return")
):
    # Cluster topics as in /api/topic_clusters
    topic_ids = list(topic_embeddings.keys())
    embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    clusters = {}
    for topic_id, label in zip(topic_ids, labels):
        clusters.setdefault(label, []).append(topic_id)
    # For each cluster, compute avg views in the latest period
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    if period == '6M':
        # Custom 6-month window: assign each date to a 6-month bin
        df['period_bin'] = (df['time_published'].dt.year * 12 + df['time_published'].dt.month - 1) // 6
        latest_period = df['period_bin'].max()
        period_mask = df['period_bin'] == latest_period
        period_label = f"6M_{latest_period}"
    else:
        df['period'] = df['time_published'].dt.to_period(period)
        latest_period = df['period'].max()
        period_mask = df['period'] == latest_period
        period_label = str(latest_period)
    trending = []
    for label, topic_list in clusters.items():
        cluster_df = df[(df['topic'].isin(topic_list)) & period_mask]
        avg_views = float(cluster_df['views'].mean()) if not cluster_df.empty else 0.0
        texts_count = int(cluster_df.shape[0])
        trending.append({
            "cluster_id": int(label),
            "topics": [int(tid) for tid in topic_list],
            "texts_count": texts_count,
            "avg_views": avg_views,
            "top_words": [
                [w for w, _ in topic_model.get_topic(tid)[:5]]
                for tid in topic_list
            ]
        })
    trending = sorted(trending, key=lambda x: x['avg_views'], reverse=True)[:top_n]
    return JSONResponse(content={
        "period": period_label,
        "clusters": trending
    })

@app.get("/api/trending_topic_clusters_moving_avg")
def get_trending_topic_clusters_moving_avg(
    distance_threshold: float = Query(0.2, description="Distance threshold for clustering (1-cosine similarity)")
):
    """
    Return trending clusters using the moving average of mean views over the last 3 non-overlapping 30-day windows.
    """
    # Cluster topics as in /api/topic_clusters
    topic_ids = list(topic_embeddings.keys())
    embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    topic_to_cluster = {int(tid): int(label) for tid, label in zip(topic_ids, labels)}
    # Assign cluster_id to all texts
    df = df_top_texts.copy()
    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df[~df['time_published'].isna()]
    df['cluster_id'] = df['topic'].map(topic_to_cluster)
    # Use the moving average helper
    trending_clusters = get_trending_clusters_moving_average(df, 'cluster_id', window_days=30, n_windows=3, n_top=5)
    # Prepare info for each trending cluster
    clusters_info = []
    for cid in trending_clusters:
        tids = [tid for tid, label in topic_to_cluster.items() if label == cid]
        cluster_df = df[df['cluster_id'] == cid]
        avg_views_trend = cluster_df['views'].mean() if not cluster_df.empty else 0
        clusters_info.append({
            "cluster_id": int(cid),
            "avg_views": float(avg_views_trend),
            "topics": [
                {
                    "topic_id": int(tid),
                    "top_words": [w for w, _ in topic_model.get_topic(tid)[:10]]
                } for tid in tids
            ]
        })
    clusters_info = sorted(clusters_info, key=lambda x: x["avg_views"], reverse=True)
    return JSONResponse(content={"clusters": clusters_info})

print(df_top_texts['time_published'].min(), df_top_texts['time_published'].max())
print(df_top_texts['time_published'].value_counts())
print(df_top_texts['time_published'].isna().sum())

# To run: uvicorn service:app --reload
# http://127.0.0.1:8000/static/dashboard.html