# Cell 1: Install dependencies (run this in Colab)
# !pip install -U spacy
# !python -m spacy download ru_core_news_lg
# !pip install nltk tqdm

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import spacy
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import nltk
# from google.colab import drive

# Download NLTK stopwords if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Enable GPU if available
try:
    spacy.require_gpu()
    print("spaCy is using GPU.")
except Exception as e:
    print("spaCy is using CPU.")

# Load spaCy Russian model
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

def spacy_lemmatize_batch(texts, batch_size=200):
    print(f"Lemmatizing texts in batches of {batch_size} using spaCy and GPU (if available)...")
    docs = nlp.pipe(texts, batch_size=batch_size)
    # Wrap docs with tqdm for progress bar
    return [" ".join([token.lemma_ for token in doc]) for doc in tqdm(docs, total=len(texts), desc="spaCy lemmatization")]

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return text

def main():
    # Example: read parquet or CSV
    input_file = 'processed_habr_data.parquet'  # or .csv
    output_folder = 'preprocessed_batches_spacy'  # Folder to save batch pickles
    batch_size = 10000  # Number of texts per batch
    if input_file.endswith('.parquet'):
        print(f"Reading parquet file: {input_file}")
        df = pd.read_parquet(input_file)
    else:
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
    # Filter out English if needed
    if 'lang' in df.columns:
        print("Filtering out English texts...")
        df = df[df['lang'] != 'en'].reset_index(drop=True)
    texts = df['text'].fillna("").tolist()
    print(f"Cleaning {len(texts)} texts (lowercase, remove punctuation/numbers)...")
    cleaned_texts = [preprocess_text(text) for text in tqdm(texts, desc="Cleaning texts")]
    # Reverse the cleaned_texts so that batching starts from the latest texts
    cleaned_texts = cleaned_texts[::-1]
    # Also reverse the corresponding dates and ids
    if 'time_published' in df.columns:
        dates = df['time_published'].fillna('').tolist()[::-1]
    else:
        dates = [''] * len(cleaned_texts)
    if 'id' in df.columns:
        ids = df['id'].tolist()[::-1]
    else:
        ids = list(range(len(cleaned_texts)))
    os.makedirs(output_folder, exist_ok=True)
    n_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    # Find already processed batches
    existing_batches = set()
    for fname in os.listdir(output_folder):
        if fname.startswith('preprocessed_texts_spacy_batch_') and fname.endswith('.pkl'):
            try:
                batch_num = int(fname.split('_batch_')[1].split('.pkl')[0])
                existing_batches.add(batch_num)
            except Exception:
                continue
    for i in range(n_batches):
        batch_num = i + 1
        batch_pkl_path = os.path.join(output_folder, f'preprocessed_texts_spacy_batch_{batch_num}.pkl')
        batch_info_path = os.path.join(output_folder, f'preprocessed_texts_spacy_batch_{batch_num}_info.txt')
        if batch_num in existing_batches:
            print(f"Skipping batch {batch_num}: already exists at {batch_pkl_path}")
            continue
        batch_texts = cleaned_texts[i*batch_size:(i+1)*batch_size]
        batch_dates = dates[i*batch_size:(i+1)*batch_size]
        batch_ids = ids[i*batch_size:(i+1)*batch_size]
        print(f"Batch {batch_num}/{n_batches}: Lemmatizing {len(batch_texts)} texts with spaCy...")
        lemmatized_texts = spacy_lemmatize_batch(batch_texts, batch_size=200)
        print(f"Batch {batch_num}: Filtering tokens (stopwords, numbers, short words)...")
        preprocessed_texts = []
        for text in tqdm(lemmatized_texts, desc=f"Filtering tokens (batch {batch_num})"):
            words = word_tokenize(text)
            processed_words = [word for word in words if not is_number(word) and word.lower() not in all_stopwords and len(word) >= 2]
            preprocessed_texts.append(" ".join(processed_words))
        # Save both ids and texts
        with open(batch_pkl_path, 'wb') as f:
            pickle.dump({'ids': batch_ids, 'texts': preprocessed_texts}, f)
        # Write batch info file
        if batch_dates and batch_dates[0] != '':
            batch_dates_parsed = pd.to_datetime(batch_dates, errors='coerce')
            min_date = batch_dates_parsed.min()
            max_date = batch_dates_parsed.max()
            with open(batch_info_path, 'w', encoding='utf-8') as f:
                f.write(f'Batch number: {batch_num}\n')
                f.write(f'Date range: {min_date} to {max_date}\n')
        else:
            with open(batch_info_path, 'w', encoding='utf-8') as f:
                f.write(f'Batch number: {batch_num}\n')
                f.write('Date range: unknown\n')
        print(f"Saved batch {batch_num} to {batch_pkl_path} and info to {batch_info_path}")
    print(f"All batches saved to folder: {output_folder}")

if __name__ == '__main__':
    main() 