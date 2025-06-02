import pandas as pd
import pickle

print("Loading the main DataFrame")
# Load the main DataFrame
df = pd.read_parquet('processed_habr_data_with_spacy_1_21.parquet')

# Load topic assignments (make sure you use the correct .pkl file)
print("Loading topic assignments")
with open('topicmodel2305/topics_colab_sber_sbert_REDUSED.pkl', 'rb') as f:
    topics = pickle.load(f)

print("Sanity check: lengths must match")
# Sanity check: lengths must match
assert len(df) == len(topics), f"Length mismatch: {len(df)} vs {len(topics)}"

print("Adding topic assignments to the DataFrame")
# Add topic assignments to the DataFrame
df['topic'] = topics

print("Saving the desired columns to a new CSV")
# Save the desired columns to a new CSV
columns_to_save = ['id', 'time_published', 'views', 'comments_count', 'text_spacy', 'text', 'topic', 'hubs']
df[columns_to_save].to_csv('texts_with_topics.csv', index=False)