# Google Colab version of topic_modeling.py for GPU acceleration
# Instructions:
# 1. Upload your data file (processed_habr_data.parquet) to Colab or mount Google Drive.
# 2. Run the pip installs below.
# 3. Adjust file paths as needed for your Colab environment.

# !pip install bertopic sentence-transformers umap-learn hdbscan pandas pyarrow tqdm nltk wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize stemmers and lemmatizer
russian_stemmer = SnowballStemmer("russian")
english_stemmer = SnowballStemmer("english")
english_lemmatizer = WordNetLemmatizer()

try:
    if spacy.prefer_gpu():
        print("spaCy: GPU is available and will be used.")
        spacy.require_gpu()
    else:
        print("spaCy: GPU is NOT available, using CPU.")
except ImportError:
    print("spaCy is not installed.")

nlp = spacy.load("ru_core_news_lg")

# Get stopwords
russian_stopwords = set(stopwords.words('russian'))
english_stopwords = set(stopwords.words('english'))
all_stopwords = russian_stopwords.union(english_stopwords)

# Add custom stopwords
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

def is_number(word):
    return bool(re.match(r'^[\d\.,]+$', word))

def spacy_lemmatize_batch(texts, batch_size=1000):
    print(f"Lemmatizing texts in batches of {batch_size} using spaCy and GPU (if available)...")
    docs = nlp.pipe(texts, batch_size=batch_size)
    return [" ".join([token.lemma_ for token in doc]) for doc in tqdm(docs, total=len(texts), desc="spaCy lemmatization")]

def filter_tokens(text):
    words = word_tokenize(text)
    return " ".join([
        word for word in words
        if not is_number(word) and word.lower() not in all_stopwords and len(word) >= 2
    ])

# Function to lemmatize input text using spaCy
def lemmatize_input_text(text, nlp=None):
    if nlp is None:
        import spacy
        nlp = spacy.load("ru_core_news_lg")
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def preprocess_single_text(text, nlp):
    # Lowercase
    text = str(text).lower()
    # Lemmatize with spaCy
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    # Filter tokens
    processed = filter_tokens(lemmatized)
    return processed

# Add this flag at the top (after imports)
USE_SAVED_EMBEDDINGS = False  # Set to True to load, False to compute and save

def main():
    # from google.colab import drive
    # drive.mount('/content/drive')
    # output_dir = '/content/drive/MyDrive/bertopic_outputs'
    # os.makedirs(output_dir, exist_ok=True)
    # df = pd.read_parquet('processed_habr_data_with_spacy_1_21.parquet')
    # texts = df['text_spacy'].fillna("").tolist()
    # print(f"Cleaning {len(texts)} texts (lowercase, remove punctuation/numbers)...")
    # cleaned_texts = [str(text).lower() for text in tqdm(texts, desc="Cleaning")]
    # print("Batch lemmatizing with spaCy...")
    # lemmatized_texts = spacy_lemmatize_batch(cleaned_texts, batch_size=1000)
    # print("Filtering tokens (stopwords, numbers, short words)...")
    # preprocessed_texts = [filter_tokens(text) for text in tqdm(lemmatized_texts, desc="Filtering tokens")]
    # with open(os.path.join(output_dir, 'preprocessed_texts.pkl'), 'wb') as f:
    #     pickle.dump(preprocessed_texts, f)

    # Use GPU for embeddings
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device='cuda')

    umap_model = UMAP(n_neighbors=30, 
                      n_components=10, 
                      min_dist=0.05, 
                      metric='cosine', 
                      random_state=42)
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=200,
        min_samples=50,
        metric='euclidean',
        # cluster_selection_method='leaf',
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(lowercase = False,
                                       stop_words=None,
                                        min_df=1, 
                                        max_df=5000, 
                                        ngram_range=(1, 3), max_features=100000)
    topic_model = BERTopic(
        # embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        languauge = 'multilingual'
        min_topic_size=8,
        nr_topics='auto',
        verbose=True,
        calculate_probabilities=True
    )

    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    if USE_SAVED_EMBEDDINGS and os.path.exists(embeddings_path):
        print(f"Loading precomputed embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    else:
        print("Computing embeddings and saving for future runs...")
        topics, probs = topic_model.fit_transform(texts)
        np.save(embeddings_path, topic_model.embeddings_)
        print(f"Saved embeddings to {embeddings_path}")

    # Outlier analysis
    topics_arr = np.array(topics)
    outlier_count = np.sum(topics_arr == -1)
    total_count = len(topics_arr)
    outlier_percentage = 100 * outlier_count / total_count
    print(f"Number of outliers: {outlier_count} out of {total_count} ({outlier_percentage:.2f}%)")


    # Evaluate topic coherence using BERTopic's built-in method
    coherence_score = topic_model.evaluate_topics(preprocessed_texts)
    print(f"Topic Coherence Score: {coherence_score:.4f}") 

    # Save topics and model if needed
    with open(os.path.join(output_dir, 'topics_colab.pkl'), 'wb') as f:
        pickle.dump(topics, f)
    topic_model.save(os.path.join(output_dir, 'bertopic_model_colab'))

    # (Optional) Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, 'topic_info_colab.csv'), index=False)

    # (Optional) Visualize topics
    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(output_dir, 'umap_topics_colab.html'))

    # --- Filter for the last 6 months ---
    df['time_published'] = pd.to_datetime(df['time_published'])
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
    df_recent = df[df['time_published'] >= cutoff].copy()

    # --- Assign topics to recent data ---
    texts_recent = [filter_tokens(text) for text in tqdm(df_recent['text_spacy'].fillna(''), desc='Filtering recent texts')]
    topics_recent, _ = topic_model.transform(texts_recent)
    df_recent['topic'] = topics_recent

    # --- Find most popular topics in the last 6 months by average views ---
    topic_stats = (
        df_recent.groupby('topic')
        .agg(
            count=('topic', 'size'),
            avg_views=('views', 'mean')
        )
        .sort_values('avg_views', ascending=False)
        .head(10)
    )

    print("\nTop 10 topics in the last 6 months by average views:")
    for topic_id, row in topic_stats.iterrows():
        print(f"Topic {topic_id}: {row['count']} articles, average views: {row['avg_views']:.1f}")
        print(f"Top words: {topic_model.get_topic(topic_id)}\n")

    # --- Output examples of the most popular texts of the last 6 months ---
    # For each of the top topics, get the top 3 most viewed articles
    output_rows = []
    for topic_id in topic_stats.index:
        top_articles = df_recent[df_recent['topic'] == topic_id].sort_values(by='views', ascending=False).head(3)
        for _, row in top_articles.iterrows():
            output_rows.append({
                'id': row.get('id', ''),
                'date': row['time_published'],
                'topic': topic_id,
                'views': row.get('views', 0),
                'comments_count': row.get('comments_count', 0),
                'text': row.get('text_spacy', '')
            })
    # Save to CSV
    pd.DataFrame(output_rows).to_csv(os.path.join(output_dir, 'top_texts_last_6_months.csv'), index=False)
    # Print examples
    print("\nExamples of the most popular texts of the last 6 months:")
    for row in output_rows:
        print(f"ID: {row['id']}")
        print(f"Date: {row['date']}")
        print(f"Topic: {row['topic']}")
        print(f"Views: {row['views']}, Comments: {row['comments_count']}")
        print(f"Text: {row['text'][:500]}...")
        print("="*80)

    # --- Topic size distribution (bar plot) ---
    topic_info = topic_model.get_topic_info()
    plt.figure(figsize=(12, 6))
    plt.bar(topic_info['Topic'].astype(str), topic_info['Count'])
    plt.title('Topic Size Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_size_distribution.png'))
    plt.close()

    # --- Top words per topic (bar charts) ---
    os.makedirs(os.path.join(output_dir, 'topic_word_bars'), exist_ok=True)
    for topic_id in topic_info['Topic']:
        if topic_id == -1:
            continue  # skip outlier
        words_scores = topic_model.get_topic(topic_id)
        if not words_scores:
            continue
        words, scores = zip(*words_scores[:10])
        plt.figure(figsize=(8, 4))
        plt.barh(words[::-1], scores[::-1], color='skyblue')
        plt.title(f'Topic {topic_id}: Top Words')
        plt.xlabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_word_bars', f'topic_{topic_id}_words.png'))
        plt.close()

    # --- Save top words per topic as a CSV table ---
    rows = []
    for topic_id in topic_info['Topic']:
        if topic_id == -1:
            continue
        words_scores = topic_model.get_topic(topic_id)
        if not words_scores:
            continue
        words = [w for w, _ in words_scores[:10]]
        rows.append({'Topic': topic_id, 'Top Words': ', '.join(words)})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'top_words_per_topic.csv'), index=False)

    # --- Enhanced Interactive cell: Check similarity of input text to top topics of last 6 months ---
    # Run this cell after running the main script
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    output_dir = '/content/drive/MyDrive/bertopic_outputs'
    df_top_texts = pd.read_csv(os.path.join(output_dir, 'top_texts_last_6_months.csv'))

    N = 10  # or any number you want
    avg_views_by_topic = df_top_texts.groupby('topic')['views'].mean().sort_values(ascending=False)
    top_topics = avg_views_by_topic.head(N).index.tolist()
    topic_rank = {topic_id: rank+1 for rank, topic_id in enumerate(top_topics)}

    # Get representative embeddings for each top topic (mean embedding of top articles)
    topic_embeddings = {}
    for topic_id in top_topics:
        topic_texts = df_top_texts[df_top_texts['topic'] == topic_id]['text'].tolist()
        if topic_texts:
            topic_embs = sentence_model.encode(topic_texts, show_progress_bar=False)
            topic_embeddings[topic_id] = np.mean(topic_embs, axis=0)

    # Function to check similarity
    user_text = input('Enter your text: ')
    user_text_processed = preprocess_single_text(user_text, nlp)
    user_emb = sentence_model.encode([user_text_processed])[0]

    # Compute similarities to all topics
    similarities = []
    for topic_id, topic_emb in topic_embeddings.items():
        sim = cosine_similarity([user_emb], [topic_emb])[0][0]
        similarities.append((topic_id, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Output the most similar topic
    best_topic, best_score = similarities[0]

    # Get average views/comments and rank for the best topic
    topic_df = df_top_texts[df_top_texts['topic'] == best_topic]
    avg_views = topic_df['views'].mean()
    avg_comments = topic_df['comments_count'].mean()
    topic_position = topic_rank.get(best_topic, None)

    # After finding best_topic, best_score
    topic_words_scores = topic_model.get_topic(best_topic)
    topic_words = [w for w, _ in topic_words_scores[:10]] if topic_words_scores else []
    topic_df = df_top_texts[df_top_texts['topic'] == best_topic]
    avg_views = topic_df['views'].mean()
    avg_comments = topic_df['comments_count'].mean()

    print(f"Most similar topic: {best_topic}")
    print(f"Cosine similarity: {best_score:.3f}")
    if best_score < 0.65:
        print("Your input is not similar to any topic (similarity < 0.65).\n")
    print(f"Average views (last 6 months): {avg_views:.1f}")
    print(f"Average comments (last 6 months): {avg_comments:.1f}")
    print(f"Top words: {', '.join(topic_words)}")

    # --- EXPLANATION ---
    lemm_input_set = set(user_text_processed.lower().split())
    matched_words = [w for w in topic_words if w in lemm_input_set]

    print("\nWhy is this topic considered similar to your input?")
    print(f"Top words of topic {best_topic}: {', '.join(topic_words)}")
    if matched_words:
        print(f"Your input shares these words with the topic: {', '.join(matched_words)}")
    else:
        print("Your input does not directly share top words with the topic, but the overall semantic meaning is similar.")

    # Brief summary of the topic's main themes
    if topic_words:
        print(f"This topic is mainly about: {', '.join(topic_words[:5])} ...")

    # Output the next most related topics (top 3 total)
    print("\nMost related topics:")
    for i, (topic_id, sim) in enumerate(similarities[:3]):
        topic_df = df_top_texts[df_top_texts['topic'] == topic_id]
        avg_views = topic_df['views'].mean()
        avg_comments = topic_df['comments_count'].mean()
        position = topic_rank.get(topic_id, None)
        print(f"#{i+1} Topic {topic_id}: similarity={sim:.3f}, avg views={avg_views:.1f}, avg comments={avg_comments:.1f}, rank={position}")

    # After topic modeling, add topic numbers as a new column to the main DataFrame
    df['topic'] = topics
    # Save the updated DataFrame with topics as a new parquet file
    df.to_parquet(os.path.join(output_dir, 'processed_habr_data_with_topics.parquet'), index=False)
    print(f"Saved DataFrame with topic numbers to {os.path.join(output_dir, 'processed_habr_data_with_topics.parquet')}")

if __name__ == '__main__':
    main() 