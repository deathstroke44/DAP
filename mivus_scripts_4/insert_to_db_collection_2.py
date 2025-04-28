import os
import time
import random
import numpy as np
from pymilvus import MilvusClient

states = [
    'Atlanta',
    'Austin',
    'Charlotte',
    'Dallas',
    'Houston',
    'LosAngeles'
]

# Connect to Milvus
client = MilvusClient("scalibility.db")

# Change working directory
os.chdir('../data/train_set/')
print("Files found:", os.listdir())

cnt = 200000  # Global ID counter

def insert_data(fold=1, seed=42, noise_std_scale=0.01):
    global cnt
    print(f"Started inserting with fold={fold}")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    X = np.load('X_train_combined.npy', allow_pickle=True)
    Y = np.load('y_train_combined.npy', allow_pickle=True)

    # Calculate feature-wise standard deviation for noise
    feature_std = np.std(X, axis=0)

    batch_data = []

    for x, y in zip(X, Y):
        for i in range(int(fold)):  # for each fold
            state = random.choice(states)
            cnt += 1

            if i == 0:
                x_aug = x  # use original in first fold
            else:
                noise = np.random.normal(loc=0.0, scale=noise_std_scale * feature_std, size=x.shape)
                x_aug = x + noise  # add Gaussian noise

            batch_data.append({
                "id": cnt,
                "vector": x_aug.tolist(),
                "state": state,
                "label": int(y[0]) if isinstance(y, (np.ndarray, list)) else int(y)
            })

    # Now insert everything at once
    res = client.insert(collection_name="collection_1", data=batch_data)
    print(f"Inserted {len(batch_data)} total records {fold}, final cnt={cnt}")

# Insert phase
start_time = time.time()
insert_data(fold='1', seed=42, noise_std_scale=0.01)  # Example: 5x data
elapsed = time.time() - start_time
print(f"Insert Execution time: {elapsed:.2f} seconds")

def evaluate():
    k = 20

    # Load test data
    X = np.load('/users/PAS2671/kabir36/ns_project/DAP/data/train_set/X_test_Atlanta.npy', allow_pickle=True)

    start_time = time.time()

    results = client.search(
        collection_name="collection_1",
        data=[x.tolist() for x in X],
        limit=k,
        output_fields=["label"],
        search_params={"nprobe": 8},
        filter="state == 'Atlanta'"
    )

    elapsed = time.time() - start_time
    print(f"Search Execution time: {elapsed:.2f} seconds")

evaluate()
