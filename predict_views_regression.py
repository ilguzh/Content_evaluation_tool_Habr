import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading data...")
# Load data
csv_file = 'texts_with_topics.csv'
df = pd.read_csv(csv_file, usecols=['views', 'hubs', 'topic', 'text', 'time_published'])

print("Extracting basic features...")
# Basic features
def extract_length(text):
    if pd.isna(text):
        return 0
    return len(str(text))

df['text_length'] = df['text'].apply(extract_length)
df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
df['year'] = df['time_published'].dt.year.fillna(0).astype(int)
df['month'] = df['time_published'].dt.month.fillna(0).astype(int)
df['day'] = df['time_published'].dt.day.fillna(0).astype(int)

print("Encoding hubs and topics...")
# One-hot encode hubs and topics
def split_list(col):
    if pd.isna(col):
        return []
    return [x.strip() for x in str(col).split(',') if x.strip()]

df['hub_list'] = df['hubs'].apply(split_list)
df['topic_list'] = df['topic'].apply(split_list)

# Prepare one-hot encoders
hub_encoder = MultiLabelBinarizer()
topic_encoder = MultiLabelBinarizer()

hub_features = hub_encoder.fit_transform(df['hub_list'])
topic_features = topic_encoder.fit_transform(df['topic_list'])

print("Preparing feature sets...")
# Feature sets
y = df['views'].values
X_basic = df[['text_length', 'year', 'month', 'day']].values
X_hubs = hub_features
X_topics = topic_features
X_both = np.hstack([hub_features, topic_features])

# Train/test split
X_sets = {
    'Basic': X_basic,
    'Hubs': X_hubs,
    'Topics': X_topics,
    'Hubs+Topics': X_both
}
results = {}
for name, X in X_sets.items():
    print(f"Training and evaluating model: {name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"--- {name} ---")
    print(f"R^2:   {r2:.4f}")
    print(f"MAE:   {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}\n")

print("\nSummary of Results:")
print(pd.DataFrame(results).T) 