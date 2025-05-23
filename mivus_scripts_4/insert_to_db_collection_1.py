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

cnt = 0  # Global ID counter

def insert_data(fold):
    global cnt
    print(f"Started inserting {fold}")

    X = np.load('X_train_combined.npy', allow_pickle=True)
    Y = np.load('y_train_combined.npy', allow_pickle=True)

    batch_data = []

    for x, y in zip(X, Y):
        state = random.choice(states)  # random state per point
        cnt += 1
        batch_data.append({
            "id": cnt,
            "vector": x.tolist(),
            "state": state,
            "label": int(y[0]) if isinstance(y, (np.ndarray, list)) else int(y)
        })

    res = client.insert(collection_name="collection_1", data=batch_data)
    print(f"Inserted {len(batch_data)} records from {fold}. Total inserted: {cnt}")

# Insert phase
start_time = time.time()
insert_data('1')
elapsed = time.time() - start_time
print(f"Insert Execution time: {elapsed:.2f} seconds")

def evaluate():
    k = 20

    # Correct file path and name
    X = np.load('/users/PAS2671/kabir36/ns_project/DAP/data/train_set/X_test_Atlanta.npy', allow_pickle=True)

    start_time = time.time()

    results = client.search(
        collection_name="collection_1",
        data=[x.tolist() for x in X],
        limit=k,
        output_fields=["label"],
        search_params={"nprobe": 8},
        filter="state == 'Atlanta'"  # fixed filter
    )

    elapsed = time.time() - start_time
    print(f"Search Execution time: {elapsed:.2f} seconds")

evaluate()
