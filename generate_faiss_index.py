#https://www.kaggle.com/datasets/gmhost/wikipedia-stem-plaintext
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
model = SentenceTransformer('thenlper/gte-base')
model.max_seq_length = 512

df = pd.read_parquet(f"/content/wikipedia/cohere.parquet", columns=['text'])
#df = pd.read_parquet(f"/content/wikipedia/parsed.parquet", columns=['text'])
contexts = list(df['text'])

import faiss
encoded_data = model.encode(contexts, batch_size=256, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
encoded_data = encoded_data.detach().cpu().numpy()
encoded_data = np.asarray(encoded_data.astype('float32'))
index = faiss.IndexFlatIP(encoded_data.shape[1])

index.add(encoded_data)
faiss.write_index(index, 'cohere_gte-base.index')
#faiss.write_index(index, 'parsed_gte-base.index')
