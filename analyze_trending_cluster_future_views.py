import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

print("Loading data...")
# --- Parameters ---
CLUSTER_THRESHOLD = 0.05
CSV_FILE = 'texts_with_topics.csv'
EMBEDDINGS_FILE = 'topicmodel2305/topic_embeddings_sber.pkl'
N_TOP = 10  # Number of trending clusters to consider
WINDOW_DAYS = 30
N_WINDOWS = 120  # Number of rolling windows to analyze

# --- Load data ---
df = pd.read_csv(CSV_FILE, usecols=['views', 'topic', 'time_published'])
df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
df = df[~df['time_published'].isna()]

print("Clustering topics ONCE for all windows...")
# --- Load and cluster topic embeddings ONCE ---
with open(EMBEDDINGS_FILE, 'rb') as f:
    topic_embeddings = pickle.load(f)
topic_ids = list(topic_embeddings.keys())
embeddings = np.stack([topic_embeddings[tid] for tid in topic_ids])
sim_matrix = cosine_similarity(embeddings)
distance_matrix = 1 - sim_matrix
clustering = AgglomerativeClustering(
    metric='precomputed',
    linkage='average',
    distance_threshold=CLUSTER_THRESHOLD,
    n_clusters=None
)
labels = clustering.fit_predict(distance_matrix)
topic_to_cluster = {str(tid): int(label) for tid, label in zip(topic_ids, labels)}

print("Assigning clusters to texts...")
def get_first_topic(topics):
    if pd.isna(topics):
        return None
    if isinstance(topics, str):
        return topics.split(',')[0].strip()
    return str(topics).strip()

df['main_topic'] = df['topic'].apply(get_first_topic)
df['cluster_id'] = df['main_topic'].map(topic_to_cluster)
df = df.dropna(subset=['cluster_id'])

def analyze_window(window2_end, window_idx, cluster_ma_dict):
    window2_start = window2_end - pd.Timedelta(days=WINDOW_DAYS)
    window1_end = window2_start
    window1_start = window1_end - pd.Timedelta(days=WINDOW_DAYS)
    prev_window_df = df[(df['time_published'] > window1_start) & (df['time_published'] <= window1_end)]
    recent_window_df = df[(df['time_published'] > window2_start) & (df['time_published'] <= window2_end)]
    print(f"\n=== Window {window_idx+1} ===")
    print(f"Previous window: {window1_start.date()} to {window1_end.date()}")
    print(f"Recent window:   {window2_start.date()} to {window2_end.date()}")

    # Check for empty recent_window_df or missing 'views'
    if recent_window_df.empty or 'views' not in recent_window_df.columns:
        print("No data in recent window. Skipping this window.")
        return None

    # --- Moving average for each cluster ---
    # Update cluster_ma_dict with this window's means
    cluster_means = prev_window_df.groupby('cluster_id')['views'].mean()
    for cid in cluster_means.index:
        if cid not in cluster_ma_dict:
            cluster_ma_dict[cid] = []
        cluster_ma_dict[cid].append(cluster_means[cid])
        # Keep only last 3 windows
        if len(cluster_ma_dict[cid]) > 3:
            cluster_ma_dict[cid] = cluster_ma_dict[cid][-3:]
    # Compute moving average for each cluster
    cluster_ma = {cid: np.mean(vals) for cid, vals in cluster_ma_dict.items() if len(vals) == 3}
    if not cluster_ma:
        print("Not enough history for moving average. Skipping this window.")
        return None
    trending_clusters = set(pd.Series(cluster_ma).sort_values(ascending=False).head(N_TOP).index)
    recent_window_df = recent_window_df.copy()
    recent_window_df['was_trending'] = recent_window_df['cluster_id'].apply(lambda x: x in trending_clusters)
    trending = recent_window_df[recent_window_df['was_trending']]
    not_trending = recent_window_df[~recent_window_df['was_trending']]

    # Check for empty or malformed trending/not_trending
    if trending.empty or not_trending.empty or 'views' not in trending.columns or 'views' not in not_trending.columns:
        print("No data in trending or not trending group. Skipping this window.")
        return None

    print('Mean views (was trending):', trending['views'].mean())
    print('Mean views (not trending):', not_trending['views'].mean())
    print(f"Trending group size: {len(trending)}")
    print(f"Not trending group size: {len(not_trending)}")
    print("Trending clusters (moving avg):", {int(x) for x in trending_clusters})
    print("Clusters present in next window:", {int(x) for x in recent_window_df['cluster_id'].unique()})
    print("Overlap:", {int(x) for x in trending_clusters & set(recent_window_df['cluster_id'].unique())})
    if len(trending) == 0 or len(not_trending) == 0 or len(trending_clusters & set(recent_window_df['cluster_id'].unique())) == 0:
        print("No overlap between trending clusters and clusters in the next window. Skipping statistical tests and plot.")
        return None
    stat, pval = ttest_ind(trending['views'], not_trending['views'], equal_var=False)
    stat_u, pval_u = mannwhitneyu(trending['views'], not_trending['views'])
    print(f'T-test: stat={stat:.2f}, p-value={pval:.3e}')
    print(f'Mann-Whitney U: stat={stat_u:.2f}, p-value={pval_u:.3e}')
    return {
        'window_idx': window_idx+1,
        'window2_start': window2_start,
        'window2_end': window2_end,
        'mean_trending': trending['views'].mean(),
        'mean_not_trending': not_trending['views'].mean(),
        'pval_ttest': pval,
        'pval_mwu': pval_u,
        'n_trending': len(trending),
        'n_not_trending': len(not_trending)
    }

# Analyze multiple rolling windows with moving average
results = []
window2_end = df['time_published'].max()
cluster_ma_dict = {}
for i in range(N_WINDOWS):
    res = analyze_window(window2_end, i, cluster_ma_dict)
    if res is not None:
        results.append(res)
    window2_end = window2_end - pd.Timedelta(days=WINDOW_DAYS)

# Plot results over time
if results:
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10,6))
    plt.plot(results_df['window_idx'], results_df['mean_trending'], label='Trending Clusters')
    plt.plot(results_df['window_idx'], results_df['mean_not_trending'], label='Not Trending Clusters')
    plt.xlabel('Window (1=most recent)')
    plt.ylabel('Mean Views')
    plt.title('Mean Views in Next Window by Trending Cluster Status (Rolling Windows)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean_views_trending_vs_not_rolling.png', dpi=150)
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(results_df['window_idx'], results_df['pval_ttest'], label='T-test p-value')
    plt.plot(results_df['window_idx'], results_df['pval_mwu'], label='Mann-Whitney U p-value')
    plt.xlabel('Window (1=most recent)')
    plt.ylabel('p-value')
    plt.title('Statistical Test p-values (Rolling Windows)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pvalues_trending_vs_not_rolling.png', dpi=150)
    plt.show()
    print("Plots saved as 'mean_views_trending_vs_not_rolling.png' and 'pvalues_trending_vs_not_rolling.png'")

    # --- Summary statistics ---
    overall_mean_trending = results_df['mean_trending'].mean()
    overall_mean_not_trending = results_df['mean_not_trending'].mean()
    n_windows = len(results_df)
    n_trending_higher = (results_df['mean_trending'] > results_df['mean_not_trending']).sum()
    frac_trending_higher = n_trending_higher / n_windows if n_windows > 0 else 0
    n_significant = (results_df['pval_ttest'] < 0.05).sum()
    frac_significant = n_significant / n_windows if n_windows > 0 else 0
    print("\n=== SUMMARY ACROSS ALL WINDOWS ===")
    print(f"Overall mean views (trending clusters):     {overall_mean_trending:.2f}")
    print(f"Overall mean views (not trending clusters): {overall_mean_not_trending:.2f}")
    print(f"Fraction of windows where trending clusters had higher mean views: {frac_trending_higher:.1%} ({n_trending_higher}/{n_windows})")
    print(f"Fraction of windows with significant difference (p<0.05): {frac_significant:.1%} ({n_significant}/{n_windows})")
    if frac_trending_higher > 0.5 and frac_significant > 0.5:
        print("Conclusion: Being in a trending topic cluster is a good predictor of having higher views in the next 30 days.")
    elif frac_trending_higher > 0.5:
        print("Conclusion: Being in a trending topic cluster is often associated with higher views in the next 30 days, but the difference is not always statistically significant.")
    else:
        print("Conclusion: Being in a trending topic cluster is not a consistent predictor of higher views in the next 30 days.")
else:
    print("No valid windows with overlap found for analysis.") 