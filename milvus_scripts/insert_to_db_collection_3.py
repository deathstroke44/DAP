import os
from pymilvus import MilvusClient
import numpy as np
import time 

# Prepare index building params
client = MilvusClient("milvus_us_accident_3.db")

# Change to directory with datasets
os.chdir('../data/train_set/')
files = os.listdir()
print("Files found:", files)

cnt = 0  # Global counter

def insert_data(dataset):
    global cnt
    print(f"Started inserting {dataset}")
    X = np.load(dataset, allow_pickle=True)
    Y = np.load(dataset.replace('X_train_', 'y_train_'), allow_pickle=True)
    state = dataset.replace('X_train_', '').replace('.npy', '')

    batch_data = []
    for x, y in zip(X, Y):
        cnt += 1
        batch_data.append({
            "id": cnt,
            "embedding": x.tolist(),  # Ensure it's a list, not np.array
            "state": state,
            "label": int(y)
        })

    # Insert once per file
    res = client.insert(collection_name="collection_3", data=batch_data)
    print(f"Inserted {len(batch_data)} records from {dataset}")
    print(res)

# Loop through and insert

start_time = time.time()
for file in files:
    if file.startswith("X_train"):
        insert_data(file)
        
elapsed = time.time() - start_time  # End timer

print(f"Execution time: {elapsed:.2f} seconds")

