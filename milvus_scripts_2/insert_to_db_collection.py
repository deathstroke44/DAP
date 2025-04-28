import os
from pymilvus import MilvusClient
import numpy as np
import time 

# Prepare index building params


# Change to directory with datasets
os.chdir('../data/train_set/')
files = os.listdir()
print("Files found:", files)


def insert_in_db(db_name):
    client = MilvusClient(db_name)
    cnt = 0 
    start_time = time.time()
    for dataset in files:
        if dataset.startswith("X_train"):
            print(f"Started inserting {dataset}")
            X = np.load(dataset, allow_pickle=True)
            Y = np.load(dataset.replace('X_train_', 'y_train_'), allow_pickle=True)
            state = dataset.replace('X_train_', '').replace('.npy', '')

            batch_data = []
            for x, y in zip(X, Y):
                cnt += 1
                batch_data.append({
                    "id": cnt,
                    "vector": x.tolist(),  # Ensure it's a list, not np.array
                    "state": state,
                    "label": int(y)
                })

            # Insert once per file
            res = client.insert(collection_name="collection_1", data=batch_data)
            print(f"Inserted {len(batch_data)} records from {dataset}")
            print(res)
            
    elapsed = time.time() - start_time  # End timer

    print(f"Execution time {db_name}: {elapsed:.2f} seconds")

