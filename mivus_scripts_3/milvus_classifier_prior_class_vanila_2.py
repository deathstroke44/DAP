import os
from pymilvus import MilvusClient, Collection, connections
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
import time 
import heapq
def getDistribution(dataset):
    mp={'Atlanta': 0.17368638808440215, 'Austin': 0.15172564438619485, 'Charlotte': 0.20642593119135627, 'Dallas': 0.10321973263038976, 'Houston': 0.11528749028749029, 'LosAngeles': 0.2150541118527639}
    return mp[dataset]


def linearScan(base, query, Y):
    K = 20
    
    
    ground_trouth = []
    ground_trouth_dist = []
    search_start_time = time.time()
    for q in range(0,query.shape[0]):
        xq = query[q]
        heap = []
        heapq.heapify(heap)
        for b in range (0,base.shape[0]):
            xb = base[b]
            minus = xb - xq
            distance = np.dot(minus.T, minus)
            heapq.heappush(heap, (-distance,Y[b]))
            if len(heap)>K:
                heapq.heappop(heap)
        res = []
        res_dist = []
        for node in heapq.nlargest(K, heap, key=None):
            res.append((-node[0],node[1]))
        ground_trouth.append(res)
    
    return ground_trouth

def evaluate(city, db, collection_name):
    # Connect to Milvus
    

    # Load test data
    X_train = np.load(f'/home/saminyeaser/DAP/data/train_set/X_train_{city}.npy', allow_pickle=True)
    X = np.load(f'/home/saminyeaser/DAP/data/train_set/X_test_{city}.npy', allow_pickle=True)
    Y = np.load(f'/home/saminyeaser/DAP/data/train_set/y_test_{city}.npy', allow_pickle=True)
    Y_train = np.load(f'/home/saminyeaser/DAP/data/train_set/y_train_{city}.npy', allow_pickle=True)
    y_true = Y.astype(int).tolist()

    # Create filter
    filter_string='state == "[city]"'.replace('[city]',city)
    start_time = time.time()
    # Perform batch search
    from pymilvus import Collection
    
    results = linearScan(X_train,X,Y_train)

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
            distance = hit[0]
            label = hit[1]
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

ef_values = [8,16]
k_values = [5,10,15,20,25,50, 75, 100]
db = "default"
collection_name = "collection_1"



cities = ['Austin']
for city in cities:
    y_true, y_pred_majority, y_pred_weighted, y_pred_prior_distribution = evaluate(city, db, collection_name)
    
    print("City:", city)
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
