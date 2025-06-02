import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import json
from sklearn.model_selection import train_test_split

# 1. Load your texts (replace with your own data loading)
import pandas as pd

print('reading parquet...')
df = pd.read_parquet('D:/prej2/processed_habr_data_with_spacy_1_21.parquet')
texts = df['text_spacy'].fillna("").tolist()  # replace 'text' with your column name

# 2. Segment long texts into <=200 word chunks
def segment_text(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

print('Segmenting texts...')
segmented_texts = [segment_text(t) for t in tqdm(texts, desc='Segmenting')]

# 3. Flatten for embedding, keep mapping to doc index
print('Flattening segments and mapping to document indices...')
all_segments = []
doc_indices = []
for i, segs in tqdm(enumerate(segmented_texts), total=len(segmented_texts), desc='Flattening'):
    all_segments.extend(segs)
    doc_indices.extend([i]*len(segs))

# 4. Embed segments using SentenceTransformers
segment_embeddings_path = 'segment_embeddings.pkl'
if os.path.exists(segment_embeddings_path):
    print(f"Loading segment embeddings from {segment_embeddings_path}...")
    with open(segment_embeddings_path, 'rb') as f:
        seg_data = pickle.load(f)
        embeddings = seg_data['embeddings']
        doc_indices = seg_data['doc_indices']
else:
    print('Embedding segments...')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(all_segments, show_progress_bar=True, batch_size=64)
    # Save segment embeddings and doc_indices for reuse
    with open(segment_embeddings_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'doc_indices': doc_indices}, f)
    print(f'Saved segment embeddings and doc_indices to {segment_embeddings_path}')

# 5. Group embeddings by document
print('Grouping embeddings by document...')
from collections import defaultdict
doc_embs = defaultdict(list)
for idx, emb in tqdm(zip(doc_indices, embeddings), total=len(embeddings), desc='Grouping'):
    doc_embs[idx].append(emb)
skipped = [i for i in range(len(texts)) if len(doc_embs[i]) == 0]
if skipped:
    print(f"Warning: Skipped {len(skipped)} documents with no valid segments/embeddings.")
doc_embs = [np.stack(doc_embs[i]) for i in range(len(texts)) if len(doc_embs[i]) > 0]

# 6. For each document: PCA, Agglomerative Clustering, select medoids
print('Extracting medoids for each document...')
def get_medoids(embs, n_clusters=8):
    if len(embs) <= n_clusters:
        return embs
    n_components = min(embs.shape[1], 32, len(embs))
    if n_components < 2:
        reduced = embs
    else:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embs)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(reduced)
    medoids = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_embs = embs[cluster_indices]
        mean_emb = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - mean_emb, axis=1)
        medoid_idx = cluster_indices[np.argmin(dists)]
        medoids.append(embs[medoid_idx])
    return np.stack(medoids)

medoid_embs = [get_medoids(embs) for embs in tqdm(doc_embs, desc='Medoid extraction')]

# 7. Pad/truncate medoid sets for batching (for autoencoder)
print('Padding/truncating medoid sets and creating masks...')
max_medoids = 8
embed_dim = embeddings.shape[1]
def pad_medoids_with_mask(meds, max_medoids=max_medoids, embed_dim=embed_dim):
    arr = np.zeros((max_medoids, embed_dim), dtype=np.float32)
    mask = np.zeros((max_medoids,), dtype=np.float32)
    n = min(len(meds), max_medoids)
    arr[:n] = meds[:n]
    mask[:n] = 1.0
    return arr, mask, n

X_masked = [pad_medoids_with_mask(m) for m in tqdm(medoid_embs, desc='Padding')]
X = np.stack([x[0] for x in X_masked])
masks = np.stack([x[1] for x in X_masked])
lengths = np.array([x[2] for x in X_masked])

# 8. Define a recurrent autoencoder using LSTM layers
class RecurrentMedoidAutoencoder(nn.Module):
    def __init__(self, embed_dim, max_medoids, latent_dim=128, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_medoids = max_medoids
        self.latent_dim = latent_dim
        self.encoder_lstm = nn.LSTM(embed_dim, latent_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.decoder_lstm = nn.LSTM(latent_dim, embed_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x: (batch, max_medoids, embed_dim)
        batch_size = x.size(0)
        # Encoder
        packed_x = x
        # Optionally, you can use pack_padded_sequence for more efficiency
        _, (h_n, _) = self.encoder_lstm(packed_x)  # h_n: (num_layers, batch, latent_dim)
        z = h_n[-1]  # (batch, latent_dim)
        # Decoder: repeat z for each time step
        z_repeated = z.unsqueeze(1).repeat(1, self.max_medoids, 1)  # (batch, max_medoids, latent_dim)
        dec_out, _ = self.decoder_lstm(z_repeated)
        out = self.output_layer(dec_out)  # (batch, max_medoids, embed_dim)
        return out, z

# 9. Prepare PyTorch Dataset and DataLoader (with masks)
class MedoidDataset(Dataset):
    def __init__(self, X, masks):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.masks[idx]

# Split into train/val
X_train, X_val, masks_train, masks_val = train_test_split(X, masks, test_size=0.1, random_state=42)
train_dataset = MedoidDataset(X_train, masks_train)
val_dataset = MedoidDataset(X_val, masks_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 10. Train the autoencoder with L1 regularization, masking, and early stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecurrentMedoidAutoencoder(embed_dim, max_medoids).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

def masked_mse_loss(recon, target, mask):
    # recon, target: (batch, max_medoids, embed_dim), mask: (batch, max_medoids)
    diff = (recon - target) ** 2
    diff = diff.sum(dim=2)  # sum over embed_dim -> (batch, max_medoids)
    masked_diff = diff * mask
    loss = masked_diff.sum() / mask.sum().clamp(min=1)
    return loss

def l1_regularization(model, l1_lambda=1e-5):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm

# Early stopping params
patience = 3
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
num_epochs = 500
l1_lambda = 1e-5

print('Training autoencoder with masking, L1 regularization, and early stopping...')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch, mask in tqdm(train_loader, desc=f'Epoch {epoch+1} [train]'):
        batch = batch.to(device)
        mask = mask.to(device)
        recon, _ = model(batch)
        loss = masked_mse_loss(recon, batch, mask)
        loss = loss + l1_regularization(model, l1_lambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    avg_train_loss = total_loss / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch, mask in tqdm(val_loader, desc=f'Epoch {epoch+1} [val]'):
            batch = batch.to(device)
            mask = mask.to(device)
            recon, _ = model(batch)
            loss = masked_mse_loss(recon, batch, mask)
            loss = loss + l1_regularization(model, l1_lambda)
            val_loss += loss.item() * batch.size(0)
    avg_val_loss = val_loss / len(val_dataset)
    print(f'Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'medoid_autoencoder.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
            break

# Load best model
model.load_state_dict(torch.load('medoid_autoencoder.pt'))
print('Loaded best autoencoder weights from medoid_autoencoder.pt')

# 11. Get document embeddings (use all data)
print('Generating document embeddings...')
all_dataset = MedoidDataset(X, masks)
model.eval()
with torch.no_grad():
    doc_embeddings = []
    for batch, mask in tqdm(DataLoader(all_dataset, batch_size=32), desc='Embedding docs'):
        batch = batch.to(device)
        _, z = model(batch)
        doc_embeddings.append(z.cpu().numpy())
    doc_embeddings = np.concatenate(doc_embeddings, axis=0)

# Save autoencoder weights
torch.save(model.state_dict(), 'medoid_autoencoder.pt')
print('Saved autoencoder weights to medoid_autoencoder.pt')

# Save parameters
params = {
    'max_medoids': max_medoids,
    'embed_dim': embed_dim,
    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
}
with open('robust_embedding_params.json', 'w') as f:
    json.dump(params, f)
print('Saved parameters to robust_embedding_params.json')

# Save embeddings
with open('super_robust_doc_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)
print('Saved robust document embeddings to super_robust_doc_embeddings.pkl.pkl')