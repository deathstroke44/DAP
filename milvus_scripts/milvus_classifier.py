import os
from pymilvus import MilvusClient
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
import time 


def evaluate(city, db, collection_name, search_params, k):
    # Connect to Milvus
    client = MilvusClient(db)

    # Load test data
    X = np.load(f'/users/PAS2671/kabir36/ns_project/DAP/data/train_set/X_test_{city}.npy', allow_pickle=True)
    Y = np.load(f'/users/PAS2671/kabir36/ns_project/DAP/data/train_set/y_test_{city}.npy', allow_pickle=True)
    y_true = Y.astype(int).tolist()

    # Create filter
    filter_string='state == "[city]"'.replace('[city]',city)
    start_time = time.time()
    # Perform batch search
    results = client.search(
        collection_name=collection_name,
        data=[x.tolist() for x in X],  # Convert all vectors to list
        limit=k,
        output_fields=["label"],
        search_params=search_params,
        filter=filter_string
    )

    y_pred = []

    # Aggregate predictions per query
    for hits in results:
        accident_vote = 0.0
        non_accident_vote = 0.0

        for hit in hits:
            distance = hit['distance']
            label = hit['entity']['label']
            weighted_vote = 1.0 / (distance + 1e-6)  # Avoid div by zero

            if label == 0:
                non_accident_vote += weighted_vote
            else:
                accident_vote += weighted_vote

        pred = int(accident_vote >= non_accident_vote)
        y_pred.append(pred)
    
    elapsed = time.time() - start_time  # End timer

    print(f"Execution time: {elapsed:.2f} seconds")
    return y_true, y_pred

ef_values = [20, 40]
k_values = [50, 100, 200]
city = "Atlanta"
db = "milvus_us_accident_1.db"
collection_name = "collection_1"

results_summary = {}

for ef in ef_values:
    search_params = {"params": {"ef": ef}}
    for k in k_values:
        y_true, y_pred = evaluate(city, db, collection_name, search_params, k)
        print(f"\n==== ef: {ef}, k: {k} ====")
        print('Database:',db,",Collection:",collection_name)
        print("Generating classification report")
        print(classification_report(y_true, y_pred, digits=4))
        
        # Store report summary by accuracy
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print("Accuracy:", acc)