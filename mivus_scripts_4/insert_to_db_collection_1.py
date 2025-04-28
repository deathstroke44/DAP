import os
from pymilvus import MilvusClient
import numpy as np
import time 
import random

states = [
    'Atlanta',
    'Austin',
    'Charlotte',
    'Dallas',
    'Houston',
    'LosAngeles'
]


# Prepare index building params
client = MilvusClient("scalibility.db")

# Change to directory with datasets
os.chdir('../data/train_set/')
files = os.listdir()
print("Files found:", files)

cnt = 0  # Global counter

def insert_data(fold):
    global cnt
    print(f"Started inserting {fold}")
    X = np.load('X_train_combined.npy', allow_pickle=True)
    Y = np.load('y_train_combined.npy', allow_pickle=True)
    state = random.choice(states)

    batch_data = []
    for x, y in zip(X, Y):
        cnt += 1
        batch_data.append({
            "id": cnt,
            "vector": x.tolist(),  # Ensure it's a list, not np.array
            "state": state,
            "label": int(y[0])
        })

    # Insert once per file
    res = client.insert(collection_name="collection_1", data=batch_data)
    print(f"Inserted {len(batch_data)} records from {fold} count {cnt}")

# Loop through and insert

insert_data('1')
        
elapsed = time.time() - start_time  # End timer

print(f"Insert Execution time: {elapsed:.2f} seconds")


def evaluate():
    # Connect to Milvus
    global client
    k=20
    # Load test data
    X = np.load(f'/users/PAS2671/kabir36/ns_project/DAP/data/train_set/X_test_Atalanta.npy', allow_pickle=True)


    start_time = time.time()
    # Perform batch search
    results = client.search(
        collection_name=collection_name,
        data=[x.tolist() for x in X],  # Convert all vectors to list
        limit=k,
        output_fields=["label"],
        search_params={"nprobe": 8},
        filter='Atalanta'
    )

    elapsed = time.time() - start_time  # End timer
    print(f"Search Execution time: {elapsed:.2f} seconds")
        
evaluate()
 

    
