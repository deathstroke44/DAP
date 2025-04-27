import os
from pymilvus import MilvusClient
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
import time 

def getDistribution(dataset):
    mp={'Atlanta': 0.17368638808440215, 'Austin': 0.15172564438619485, 'Charlotte': 0.20642593119135627, 'Dallas': 0.10321973263038976, 'Houston': 0.11528749028749029, 'LosAngeles': 0.2150541118527639}
    return mp[dataset]

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

    y_pred_majority = []
    y_pred_weighted = []
    y_pred_prior_distribution = []

    # Aggregate predictions per query
    for hits in results:
        accident_vote = 0.0
        non_accident_vote = 0.0
        accident_vote_weighted = 0.0
        non_accident_vote_weighted = 0.0
        accident_vote_prior_distribution = 0.0
        non_accident_vote_prior_distribution = 0.0

        for hit in hits:
            distance = hit['distance']
            label = hit['entity']['label']
            weighted_vote = 1.0 / (distance + 1e-6)  # Avoid div by zero

            if label == 0:
                non_accident_vote += 1
                non_accident_vote_weighted += weighted_vote
                non_accident_vote_prior_distribution += getDistribution(city)
            else:
                accident_vote += 1
                accident_vote_weighted += weighted_vote
                accident_vote_prior_distribution += 1 - getDistribution(city)

        y_pred_majority.append(int(accident_vote >= non_accident_vote))
        y_pred_weighted.append(int(accident_vote_weighted >= non_accident_vote_weighted))
        y_pred_prior_distribution.append(int(accident_vote_prior_distribution >= non_accident_vote_prior_distribution))
    
    elapsed = time.time() - start_time  # End timer

    print(f"Execution time: {elapsed:.2f} seconds")
    return y_true, y_pred_majority, y_pred_weighted, y_pred_prior_distribution

ef_values = [20, 40, 60, 80, 100]
k_values = [5,10,15,20,25,50, 75, 100, 150, 200]
db = "milvus_us_accident_6.db"
collection_name = "collection_6"



cities = ['Atlanta', 'Austin', 'Charlotte', 'Dallas', 'Houston', 'LosAngeles']
for city in cities:
    for ef in ef_values:
        search_params = {"params": {"ef": ef}}
        for k in k_values:
            y_true, y_pred_majority, y_pred_weighted, y_pred_prior_distribution = evaluate(city, db, collection_name, search_params, k)
            print(f"\n==== ef: {ef}, k: {k} ====")
            print('Database:',db,",Collection:",collection_name, "City:", city)
            print("Generating classification report majority voting")
            print(classification_report(y_true, y_pred_majority, digits=4))
            # Store report summary by accuracy
            acc = np.mean(np.array(y_true) == np.array(y_pred_majority))
            print("Accuracy:", acc)
            print("Generating classification report weighted")
            print(classification_report(y_true, y_pred_weighted, digits=4))
            # Store report summary by accuracy
            acc = np.mean(np.array(y_true) == np.array(y_pred_weighted))
            print("Accuracy:", acc)
            print("Generating classification report prior distribution")
            print(classification_report(y_true, y_pred_prior_distribution, digits=4))
            # Store report summary by accuracy
            acc = np.mean(np.array(y_true) == np.array(y_pred_prior_distribution))
            print("Accuracy:", acc)
            

