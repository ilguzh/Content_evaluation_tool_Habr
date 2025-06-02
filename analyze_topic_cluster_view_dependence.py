import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from datetime import datetime, timedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# --- Parameters ---
CLUSTER_THRESHOLD = 0.05
CSV_FILE = 'texts_with_topics.csv'
EMBEDDINGS_FILE = 'topicmodel2305/topic_embeddings_sber.pkl'
N_RANDOM = 10  # Number of randomizations

# --- Load data ---
df = pd.read_csv(CSV_FILE, usecols=['views', 'topic', 'time_published'])

# Parse time_published
if not np.issubdtype(df['time_published'].dtype, np.datetime64):
    df['time_published'] = pd.to_datetime(df['time_published'])

# --- Load and cluster topic embeddings (service.py style) ---
with open(EMBEDDINGS_FILE, 'rb') as f:
    topic_embeddings = pickle.load(f)

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
    distance_threshold=CLUSTER_THRESHOLD,
    n_clusters=None
)
labels = clustering.fit_predict(distance_matrix)
# Build topic_id to cluster_id mapping
topic_to_cluster = {str(tid): int(label) for tid, label in zip(topic_ids, labels)}

# Assign each text to a cluster (use first topic if multiple)
def get_first_topic(topics):
    if pd.isna(topics):
        return None
    if isinstance(topics, str):
        return topics.split(',')[0].strip()
    return str(topics).strip()

def analyze_period(df_period, label):
    df_period = df_period.copy()
    df_period['main_topic'] = df_period['topic'].apply(get_first_topic)
    df_period['cluster_id'] = df_period['main_topic'].map(topic_to_cluster)
    df_period = df_period.dropna(subset=['cluster_id'])
    # Real clusters
    cluster_groups = [g['views'].values for _, g in df_period.groupby('cluster_id') if len(g) > 5]
    if len(cluster_groups) < 2:
        print(f"Not enough clusters for {label}.")
        return
    stat, pval = kruskal(*cluster_groups)
    print(f"Kruskal-Wallis H-test (topic clusters, {label}): H={stat:.2f}, p-value={pval:.3e}")
    if pval < 0.05:
        print(f"There is a statistically significant difference in views between topic clusters (threshold=0.05) in {label}.")
    else:
        print(f"No statistically significant difference in views between topic clusters (threshold=0.05) in {label}.")
    # Random clusters
    np.random.seed(42)
    cluster_sizes = [len(g) for _, g in df_period.groupby('cluster_id') if len(g) > 5]
    n_clusters = len(cluster_sizes)
    all_indices = np.arange(len(df_period))
    random_pvals = []
    for i in range(N_RANDOM):
        np.random.shuffle(all_indices)
        start = 0
        random_groups = []
        for size in cluster_sizes:
            group = df_period.iloc[all_indices[start:start+size]]['views'].values
            random_groups.append(group)
            start += size
        if len(random_groups) > 1:
            stat_r, pval_r = kruskal(*random_groups)
            random_pvals.append(pval_r)
        else:
            random_pvals.append(np.nan)
    random_pvals = np.array(random_pvals)
    print(f"Random clusters ({label}): mean p-value = {np.nanmean(random_pvals):.3e}, std = {np.nanstd(random_pvals):.3e}, min = {np.nanmin(random_pvals):.3e}, max = {np.nanmax(random_pvals):.3e}")
    if pval < np.nanmean(random_pvals):
        print(f"Topic clusters show a stronger and more significant difference in views than random clusters in {label}.")
    else:
        print(f"Random clusters show as much or more difference in views as topic clusters in {label} (unexpected).")
    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='views', data=df_period, showfliers=False)
    plt.title(f'Views by Topic Cluster ({label}, threshold=0.05)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Views')
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.tight_layout(pad=2)
    fname = f'cluster_views_{label.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Boxplot saved as '{fname}'")

# --- Last month ---
date_max = df['time_published'].max()
date_min = date_max - pd.DateOffset(days=30)
df_last_month = df[(df['time_published'] >= date_min) & (df['time_published'] <= date_max)]
analyze_period(df_last_month, 'Last Month')

# --- Last year ---
date_min_year = date_max - pd.DateOffset(days=365)
df_last_year = df[(df['time_published'] >= date_min_year) & (df['time_published'] <= date_max)]
analyze_period(df_last_year, 'Last Year')

# --- Overall ---
analyze_period(df, 'Overall') 